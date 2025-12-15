import os
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel


def load_vectorstore(
    persist_directory: str,
    embedding_model_name: str = "BAAI/bge-m3"
) -> Chroma:
    """
    ChromaDB 벡터 저장소를 로드합니다.
    
    Args:
        persist_directory: 저장 디렉토리
        embedding_model_name: 임베딩 모델 이름
        
    Returns:
        ChromaDB 벡터 저장소
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vectorstore


def load_llm(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None
) -> Optional[BaseLanguageModel]:
    """
    LLM 모델을 로드합니다.
    
    Args:
        model: 모델 이름
        api_key: OpenAI API 키 (None이면 환경변수에서 가져옴)
        
    Returns:
        LLM 모델 또는 None (API 키가 없을 경우)
    """
    if api_key is None:
        api_key = os.getenv("GPT_API_KEY")
    
    if not api_key:
        return None
    
    llm = ChatOpenAI(
        model=model,
        api_key=api_key
    )
    
    return llm

