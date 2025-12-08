import os
import re
import json
from typing import List, Optional, Tuple, Dict, Any
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from tqdm import tqdm


class TableProcessor:
    """HTML에서 표를 추출하고, 주변 컨텍스트를 찾아 LLM으로 요약하는 클래스"""
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        """
        Args:
            llm: LLM 모델 (Gemini 등). None이면 간단한 요약 방식 사용
        """
        self.llm = llm
    
    def _extract_numbers_from_text(self, text: str) -> set:
        """텍스트에서 숫자를 추출하여 set으로 반환합니다."""
        number_patterns = re.findall(r'\d+[,\.]?\d*', text)
        numbers = set()
        for num_str in number_patterns:
            cleaned = num_str.replace(',', '').replace('.', '')
            try:
                if '.' in num_str:
                    num = int(float(num_str.replace(',', '')))
                else:
                    num = int(cleaned)
                numbers.add(num)
            except (ValueError, OverflowError):
                continue
        return numbers
    
    def _parse_table_numbers(self, table_element) -> List[int]:
        """표에서 숫자를 추출하여 리스트로 반환합니다."""
        numbers = []
        rows = table_element.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            for cell in cells:
                cell_text = cell.get_text(strip=True)
                number_patterns = re.findall(r'\d+[,\.]?\d*', cell_text)
                for num_str in number_patterns:
                    try:
                        if '.' in num_str:
                            num = int(float(num_str.replace(',', '')))
                        else:
                            num = int(num_str.replace(',', ''))
                        numbers.append(num)
                    except (ValueError, OverflowError):
                        continue
        return numbers
    
    def _parse_table_headers(self, table_element) -> List[str]:
        """표의 헤더와 첫 번째 열을 추출하여 리스트로 반환합니다."""
        headers = []
        rows = table_element.find_all('tr')
        if not rows:
            return headers
        
        header_cells = rows[0].find_all(['th', 'td'])
        for cell in header_cells:
            cell_text = cell.get_text(strip=True)
            if cell_text:
                headers.append(cell_text)
        
        for row in rows[1:]:
            first_cell = row.find(['th', 'td'])
            if first_cell:
                cell_text = first_cell.get_text(strip=True)
                if cell_text:
                    headers.append(cell_text)
        
        return headers
    
    def find_context_in_pdf_by_table_structure(
        self,
        pdf_documents: List[Document],
        table_element
    ) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """
        PDF에서 표와 가장 매칭이 많이 되는 페이지를 찾습니다.
        
        Args:
            pdf_documents: PDF에서 로드한 문서 리스트
            table_element: BeautifulSoup의 table 요소
            
        Returns:
            (best_page_idx, page_content, before_text): 가장 매칭이 많은 페이지 인덱스, 내용, 헤더 앞의 문장
        """
        if not pdf_documents or not table_element:
            return None, None, None
        
        table_numbers = self._parse_table_numbers(table_element)
        table_headers = self._parse_table_headers(table_element)
        
        if not table_numbers and not table_headers:
            return None, None, None
        
        best_page_idx = None
        best_score = 0
        
        for doc_idx, doc in enumerate(pdf_documents):
            page_text = doc.page_content
            score = 0
            
            if table_numbers:
                pdf_numbers = self._extract_numbers_from_text(page_text)
                number_overlap = len(set(table_numbers) & pdf_numbers)
                score += number_overlap
            
            if table_headers:
                header_matches = 0
                for header in table_headers:
                    if header.lower() in page_text.lower():
                        header_matches += 1
                score += header_matches
            
            if score > best_score:
                best_score = score
                best_page_idx = doc_idx
        
        if best_page_idx is not None:
            page_content = pdf_documents[best_page_idx].page_content
            before_text = None
            
            if table_headers:
                header_positions = []
                for header in table_headers:
                    header_lower = header.lower()
                    page_lower = page_content.lower()
                    pos = page_lower.find(header_lower)
                    if pos != -1:
                        header_positions.append(pos)
                
                if header_positions:
                    first_header_pos = min(header_positions)
                    search_start = max(0, first_header_pos - 2000)
                    search_text = page_content[search_start:first_header_pos]
                    
                    pattern = r'[^.!?]*(?:\):|\(unaudited\))'
                    matches = list(re.finditer(pattern, search_text, re.IGNORECASE))
                    
                    if matches:
                        match = matches[-1]
                        before_text = match.group(0).strip()
                        before_text = re.sub(r'\s+', ' ', before_text).strip()
            
            return best_page_idx, page_content, before_text
        
        return None, None, None
    
    def extract_table_structure(self, table_element) -> Tuple[str, str]:
        """
        표의 구조를 추출합니다 (헤더와 첫 번째 열).
        
        Args:
            table_element: BeautifulSoup의 table 요소
            
        Returns:
            (header_row, first_column): 헤더 행과 첫 번째 열의 텍스트
        """
        rows = table_element.find_all('tr')
        if not rows:
            return "", ""
        
        # 헤더 행 추출 (첫 번째 행)
        header_cells = rows[0].find_all(['th', 'td'])
        header_row = ' | '.join([cell.get_text(strip=True) for cell in header_cells if cell.get_text(strip=True)])
        
        # 첫 번째 열 추출 (각 행의 첫 번째 셀)
        first_column_cells = []
        for row in rows[1:]:  # 헤더 제외
            first_cell = row.find(['th', 'td'])
            if first_cell:
                cell_text = first_cell.get_text(strip=True)
                if cell_text:
                    first_column_cells.append(cell_text)
        
        first_column = '\n'.join(first_column_cells[:10])  # 최대 10개 행만
        
        return header_row, first_column
    
    def extract_table_numbers(self, table_element) -> set:
        """
        HTML 표에서 숫자를 추출하여 set으로 반환합니다.
        
        Args:
            table_element: BeautifulSoup의 table 요소
            
        Returns:
            숫자 set
        """
        rows = table_element.find_all('tr')
        if not rows:
            return set()
        
        numbers = set()
        for row in rows:
            cells = row.find_all(['td', 'th'])
            for cell in cells:
                cell_text = cell.get_text(strip=True)
                number_patterns = re.findall(r'\d+[,\.]?\d*', cell_text)
                for num_str in number_patterns:
                    cleaned = num_str.replace(',', '').replace('.', '')
                    try:
                        if '.' in num_str:
                            num = int(float(num_str.replace(',', '')))
                        else:
                            num = int(cleaned)
                        if 3 <= num <= 999999999:
                            numbers.add(num)
                    except (ValueError, OverflowError):
                        continue
        
        return numbers
    
    def summarize_table_with_llm(
        self, 
        context_before: List[str], 
        table_header: str,
        first_column: str
    ) -> Tuple[str, str]:
        """
        LLM을 사용하여 표를 요약합니다.
        
        Args:
            context_before: 표 앞의 컨텍스트
            table_header: 표의 헤더 행
            first_column: 표의 첫 번째 열
            
        Returns:
            (요약 텍스트, 프롬프트) 튜플
        """
        context_text = ""
        
        if context_before:
            context_text += "표 앞 설명:\n" + "\n".join(context_before) + "\n\n"
        
        if table_header:
            context_text += "표 헤더 (맨 위 컬럼):\n" + table_header + "\n\n"
        
        if first_column:
            context_text += "표 첫 번째 열 (맨 왼쪽):\n" + first_column + "\n\n"
        
        prompt = f"""Based on the following description and table structure information, please provide a concise summary of this table's key content.

{context_text}Summary (explain the main content and meaning of the table in 2-3 sentences):"""

        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                result = response.content
            elif hasattr(response, 'text'):
                result = response.text
            else:
                result = str(response)
            
            if not result or not result.strip():
                print(f"LLM 응답이 비어있음: {type(response)}")
                return "", prompt
            
            return result.strip(), prompt
        except Exception as e:
            print(f"LLM 요약 실패: {e}")
            import traceback
            traceback.print_exc()
            return "", prompt
    
    def summarize_table_simple(self, table_content: str) -> str:
        """
        간단한 방식으로 표를 요약합니다 (LLM 없이).
        
        Args:
            table_content: 표의 원본 텍스트
            
        Returns:
            표의 요약 텍스트 (헤더 + 처음 몇 행)
        """
        lines = table_content.split('\n')
        if not lines:
            return table_content
        
        header = lines[0] if lines else ""
        
        key_rows = []
        for line in lines[1:6]:
            if line.strip():
                key_rows.append(line)
        
        summary = f"{header}\n" + "\n".join(key_rows[:5])
        if len(lines) > 6:
            summary += f"\n... (총 {len(lines)-1}개 행)"
        
        return summary
    
    def extract_tables_from_html(
        self, 
        html_path: str, 
        pdf_documents: Optional[List[Document]] = None
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        HTML 파일에서 표를 추출하고 요약하여 Document 리스트로 반환합니다.
        
        Args:
            html_path: HTML 파일 경로
            pdf_documents: PDF에서 로드한 문서 리스트 (사용하지 않음, 호환성을 위해 유지)
            
        Returns:
            (table_docs, table_mapping): 표 Document 리스트와 매핑 정보
        """
        with open(html_path, 'r', encoding='latin-1') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 불필요한 요소 제거
        for element in soup(['script', 'style', 'noscript', 'meta', 'link']):
            element.decompose()
        
        table_docs = []
        table_mapping = {}
        
        tables = soup.find_all('table')
        
        for idx, table in enumerate(tqdm(tables, desc="표 처리 중", unit="표")):
            table_text = []
            
            # Caption 추출
            caption = table.find('caption')
            caption_text = ""
            if caption:
                caption_text = caption.get_text(strip=True)
                table_text.append(caption_text)
            
            # 표의 각 행 추출
            for row in table.find_all('tr'):
                cells = []
                for cell in row.find_all(['td', 'th']):
                    cell_text = cell.get_text(separator=' | ', strip=True)
                    if cell_text:
                        cells.append(cell_text)
                if cells:
                    table_text.append(' | '.join(cells))
            
            if table_text:
                row_count = len(table_text) - (1 if caption_text else 0)
                if row_count <= 1:
                    tqdm.write(f"표 {idx}: 1줄짜리 표로 건너뜀 (row_count: {row_count})")
                    continue
                
                original_table = '\n'.join(table_text)
                
                # 표 구조 추출 (헤더와 첫 번째 열)
                table_header_row, first_column = self.extract_table_structure(table)
                
                # PDF에서 표와 가장 매칭이 많은 페이지 찾기
                matched_page_idx = None
                matched_page_content = None
                before_text = None
                if pdf_documents:
                    matched_page_idx, matched_page_content, before_text = self.find_context_in_pdf_by_table_structure(
                        pdf_documents,
                        table
                    )
                
                context_before = []
                if before_text:
                    context_before.append(before_text)
                
                llm_description = ""
                llm_prompt = ""
                if self.llm:
                    try:
                        llm_description, llm_prompt = self.summarize_table_with_llm(
                            context_before, 
                            table_header_row,
                            first_column
                        )
                        if not llm_description:
                            tqdm.write(f"표 {idx}: LLM 설명이 비어있음 (context_before: {len(context_before)}, header: {bool(table_header_row)}, first_column: {bool(first_column)})")
                    except Exception as e:
                        tqdm.write(f"표 {idx}: LLM 설명 생성 중 오류: {e}")
                else:
                    tqdm.write(f"표 {idx}: LLM이 없어서 설명을 생성하지 않음")
                
                table_summary = llm_description if llm_description else self.summarize_table_simple(original_table)
                
                table_id = f"table_{idx}"
                table_mapping[table_id] = {
                    'original': original_table,
                    'caption': caption_text,
                    'row_count': len(table_text) - 1,
                    'context_before': context_before,
                    'llm_description': llm_description,
                    'llm_prompt': llm_prompt
                }
                
                table_docs.append(Document(
                    page_content=table_summary,
                    metadata={
                        'source': html_path,
                        'type': 'table',
                        'table_id': table_id,
                        'original_table': original_table,
                        'caption': caption_text,
                        'row_count': len(table_text) - 1
                    }
                ))
        
        return table_docs, table_mapping


if __name__ == "__main__":
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    
    load_dotenv()
    
    # LLM 설정
    gpt_api_key = os.getenv("GPT_API_KEY")
    llm = None
    
    if gpt_api_key:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=gpt_api_key
        )
        print("GPT LLM 초기화 완료")
    else:
        print("GPT_API_KEY가 없어서 간단한 요약 방식 사용")
    
    # 파일 경로
    pdf_path = "data/tsla-20250930-gen.pdf"
    html_path = "data/tsla-20250930.html"
    
    print("="*60)
    print("1. PDF 파일 로드")
    print("="*60)
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_documents = pdf_loader.load()
    
    for doc in pdf_documents:
        doc.page_content = re.sub(r'\s+', ' ', doc.page_content)
    
    print(f"PDF 파일 로드 완료: {pdf_path}")
    print(f"PDF 문서 수: {len(pdf_documents)}")
    print(f"PDF 텍스트 길이: {sum([len(doc.page_content) for doc in pdf_documents])} 문자")
    
    print("\n" + "="*60)
    print("2. HTML 파일에서 표 추출 및 LLM 요약")
    print("="*60)
    
    # TableProcessor 인스턴스 생성
    processor = TableProcessor(llm=llm)
    
    html_tables, table_mapping = processor.extract_tables_from_html(html_path, pdf_documents)
    
    print(f"HTML에서 추출된 표 개수: {len(html_tables)}")
    if html_tables:
        print(f"표 요약 총 길이: {sum([len(doc.page_content) for doc in html_tables])} 문자")
        print(f"표 원본 총 길이: {sum([len(doc.metadata.get('original_table', '')) for doc in html_tables])} 문자")
        print(f"\n첫 번째 표 요약 (LLM 생성, 임베딩용):")
        print(html_tables[0].page_content[:500])
        print(f"\n첫 번째 표 원본 (metadata에 저장):")
        print(html_tables[0].metadata.get('original_table', '')[:500])
    
    print("\n" + "="*60)
    print("3. HTML 표 저장")
    print("="*60)
    
    os.makedirs("output", exist_ok=True)
    output_path = "output/table_descriptions.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(table_mapping, f, ensure_ascii=False, indent=2)
    
    llm_descriptions_count = sum(1 for v in table_mapping.values() if v.get('llm_description'))
    print(f"표별 설명 저장 완료: {output_path}")
    print(f"LLM으로 생성된 설명 개수: {llm_descriptions_count}/{len(table_mapping)}")
    print(f"저장된 표 개수: {len(html_tables)}")

