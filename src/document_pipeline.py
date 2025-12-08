import os
import re
import json
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class DocumentPipeline:
    """문서 로드, 청킹, 임베딩, 벡터 저장소 생성을 담당하는 클래스"""
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Args:
            embedding_model_name: HuggingFace 임베딩 모델 이름
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩 크기
        """
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name
        )
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_tables_from_json(self, json_path: str) -> List[Document]:
        """table_descriptions.json 파일을 로드하여 표 Document 리스트를 생성합니다."""
        with open(json_path, 'r', encoding='utf-8') as f:
            table_data = json.load(f)
        
        table_docs = []
        
        for table_id, table_info in table_data.items():
            original_table = table_info.get('original', '')
            caption = table_info.get('caption', '')
            row_count = table_info.get('row_count', 0)
            context_before = table_info.get('context_before', [])
            llm_description = table_info.get('llm_description', '')
            
            if not original_table:
                continue
            
            page_content = llm_description if llm_description else original_table
            
            table_docs.append(Document(
                page_content=page_content,
                metadata={
                    'source': json_path,
                    'type': 'table',
                    'table_id': table_id,
                    'original_table': original_table,
                    'caption': caption,
                    'row_count': row_count,
                    'context_before': context_before
                }
            ))
        
        return table_docs
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """PDF 파일을 로드합니다."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        for doc in documents:
            doc.page_content = re.sub(r'\s+', ' ', doc.page_content)
        
        return documents
    
    def load_documents(
        self,
        pdf_path: Optional[str] = None,
        table_json_path: Optional[str] = None
    ) -> List[Document]:
        """
        PDF와 표 JSON 파일을 로드하여 결합합니다.
        
        Args:
            pdf_path: PDF 파일 경로
            table_json_path: 표 정보 JSON 파일 경로
            
        Returns:
            결합된 Document 리스트
        """
        documents = []
        
        if pdf_path:
            pdf_docs = self.load_pdf(pdf_path)
            documents.extend(pdf_docs)
        
        if table_json_path:
            table_docs = self.load_tables_from_json(table_json_path)
            documents.extend(table_docs)
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        문서를 청킹합니다. 표 문서는 그대로 유지하고, 일반 문서만 청킹합니다.
        
        Args:
            documents: 청킹할 Document 리스트
            
        Returns:
            청킹된 Document 리스트
        """
        pdf_docs = [doc for doc in documents if doc.metadata.get('type') != 'table']
        table_docs = [doc for doc in documents if doc.metadata.get('type') == 'table']
        
        pdf_chunks = self.splitter.split_documents(pdf_docs)
        
        all_chunks = pdf_chunks + table_docs
        
        return all_chunks
    
    def create_chroma_store(
        self,
        documents: List[Document],
        persist_directory: str = "./chroma_db"
    ) -> Chroma:
        """
        Chroma 벡터 저장소를 생성합니다.
        
        Args:
            documents: 저장할 Document 리스트
            persist_directory: 저장 디렉토리
            
        Returns:
            Chroma 벡터 저장소 객체
        """
        db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        return db
    
    def create_faiss_store(
        self,
        documents: List[Document],
        save_directory: str = "./faiss_db"
    ) -> FAISS:
        """
        FAISS 벡터 저장소를 생성합니다.
        
        Args:
            documents: 저장할 Document 리스트
            save_directory: 저장 디렉토리
            
        Returns:
            FAISS 벡터 저장소 객체
        """
        db = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        db.save_local(save_directory)
        return db
    
    def load_chroma_store(
        self,
        persist_directory: str = "./chroma_db"
    ) -> Chroma:
        """
        기존 Chroma 벡터 저장소를 로드합니다.
        
        Args:
            persist_directory: 저장 디렉토리
            
        Returns:
            Chroma 벡터 저장소 객체
        """
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        return db
    
    def load_faiss_store(
        self,
        save_directory: str = "./faiss_db"
    ) -> FAISS:
        """
        기존 FAISS 벡터 저장소를 로드합니다.
        
        Args:
            save_directory: 저장 디렉토리
            
        Returns:
            FAISS 벡터 저장소 객체
        """
        db = FAISS.load_local(
            save_directory,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        return db

