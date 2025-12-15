import os
import json
from typing import List, Dict, Any, Optional


class Storage:
    
    def __init__(self, storage_path: str = "data/processed"):
        """
        Args:
            storage_path: JSON 파일 경로
        """
        self.storage_path = storage_path
        self.storage_dir = os.path.dirname(storage_path)
        if self.storage_dir:
            os.makedirs(self.storage_dir, exist_ok=True)
    
    def save_tables(self, tables: List[Dict[str, Any]]):
        """
        테이블 리스트를 JSON 파일로 저장합니다.
        
        Args:
            tables: 테이블 정보 리스트
        """
        data = {
            "tables": tables
        }

        tables_path = os.path.join(self.storage_path, "tables.json")
        
        try:
            with open(tables_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"저장 실패: {e}")
    
    def load_tables(self) -> List[Dict[str, Any]]:
        """
        JSON 파일에서 테이블 리스트를 로드합니다.
        
        Returns:
            테이블 정보 리스트
        """
        if not os.path.exists(self.storage_path):
            return []

        tables_path = os.path.join(self.storage_path, "tables.json")
        
        try:
            with open(tables_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('tables', [])
        except Exception as e:
            print(f"로드 실패: {e}")
            return []
    
    def get_table(self, table_id: str) -> Optional[Dict[str, Any]]:
        """
        특정 테이블을 조회합니다.
        
        Args:
            table_id: 테이블 ID
            
        Returns:
            테이블 정보 dict 또는 None
        """
        tables = self.load_tables()
        for table in tables:
            if table.get('table_id') == table_id:
                return table
        return None
    
    def get_all_tables(self) -> Dict[str, Dict[str, Any]]:
        """
        모든 테이블을 dict 형태로 반환합니다.
        
        Returns:
            {table_id: table_data} 형태의 dict
        """
        tables = self.load_tables()
        return {table.get('table_id'): table for table in tables}
    
    def save_texts(self, texts: List[Dict[str, Any]]):
        """
        텍스트 리스트를 JSON 파일로 저장합니다.
        
        Args:
            texts: 텍스트 정보 리스트
        """
        data = {
            "texts": texts
        }
        
        text_storage_path = os.path.join(self.storage_path, "texts.json")
        
        try:
            with open(text_storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"저장 실패: {e}")
    
    def load_texts(self) -> List[Dict[str, Any]]:
        """
        JSON 파일에서 텍스트 리스트를 로드합니다.
        
        Returns:
            텍스트 정보 리스트
        """
        text_storage_path = os.path.join(self.storage_path, "texts.json")
        
        if not os.path.exists(text_storage_path):
            return []
        
        try:
            with open(text_storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('texts', [])
        except Exception as e:
            print(f"로드 실패: {e}")
            return []
    
    def get_text(self, text_id: str) -> Optional[Dict[str, Any]]:
        """
        특정 텍스트를 조회합니다.
        
        Args:
            text_id: 텍스트 ID
            
        Returns:
            텍스트 정보 dict 또는 None
        """
        texts = self.load_texts()
        for text in texts:
            if text.get('text_id') == text_id:
                return text
        return None

