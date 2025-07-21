#!/usr/bin/env python3
"""
RAG Workshop - Main Application
A comprehensive RAG implementation using Redis as vector database and OpenAI for generation.

This workshop demonstrates:
1. Document processing and chunking
2. Embedding generation using sentence transformers
3. Vector storage and similarity search with Redis
4. Retrieval-augmented generation with OpenAI
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Core libraries
import redis
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

# Document processing
import requests
from bs4 import BeautifulSoup
import PyPDF2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text with metadata"""
    id: str
    text: str
    source: str
    chunk_index: int
    embedding: Optional[np.ndarray] = None

class DocumentProcessor:
    """Handles processing of different document types"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_wikipedia_url(self, url: str) -> List[DocumentChunk]:
        """Extract and chunk text from a Wikipedia URL"""
        logger.info(f"Processing Wikipedia URL: {url}")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract main content (Wikipedia specific)
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if content_div:
                text = content_div.get_text()
            else:
                text = soup.get_text()
            
            # Clean up text
            text = ' '.join(text.split())
            
            return self._chunk_text(text, url)
            
        except Exception as e:
            logger.error(f"Error processing Wikipedia URL: {e}")
            raise
    
    def process_text_file(self, file_path: str) -> List[DocumentChunk]:
        """Process a text file"""
        logger.info(f"Processing text file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            return self._chunk_text(text, file_path)
            
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            raise
    
    def process_pdf_file(self, file_path: str) -> List[DocumentChunk]:
        """Process a PDF file"""
        logger.info(f"Processing PDF file: {file_path}")
        
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            return self._chunk_text(text, file_path)
            
        except Exception as e:
            logger.error(f"Error processing PDF file: {e}")
            raise
    
    def _chunk_text(self, text: str, source: str) -> List[DocumentChunk]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + self.chunk_size // 2, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = DocumentChunk(
                    id=f"{source}_{chunk_id}",
                    text=chunk_text,
                    source=source,
                    chunk_index=chunk_id
                )
                chunks.append(chunk)
                chunk_id += 1
            
            start = end - self.chunk_overlap
        
        logger.info(f"Created {len(chunks)} chunks from {source}")
        return chunks

class EmbeddingGenerator:
    """Generates embeddings using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for document chunks"""
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a query"""
        return self.model.encode([query])[0]

class RedisVectorStore:
    """Redis-based vector storage with similarity search"""
    
    def __init__(self, redis_url: str, index_name: str = "documents"):
        self.redis_client = redis.from_url(redis_url)
        self.index_name = index_name
        self.embedding_dim = None
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info("✅ Connected to Redis successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    def create_index(self, embedding_dim: int):
        """Create Redis search index for vectors"""
        self.embedding_dim = embedding_dim
        
        try:
            # Check if index already exists
            self.redis_client.ft(self.index_name).info()
            logger.info(f"Index '{self.index_name}' already exists")
            return
        except:
            pass
        
        # Create index schema
        from redis.commands.search.field import VectorField, TextField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        
        schema = [
            TextField("text"),
            TextField("source"),
            TextField("chunk_id"),
            VectorField("embedding", "HNSW", {
                "TYPE": "FLOAT32",
                "DIM": embedding_dim,
                "DISTANCE_METRIC": "COSINE"
            })
        ]
        
        definition = IndexDefinition(prefix=[f"{self.index_name}:"], index_type=IndexType.HASH)
        
        self.redis_client.ft(self.index_name).create_index(schema, definition=definition)
        logger.info(f"✅ Created Redis index '{self.index_name}' with dimension {embedding_dim}")
    
    def store_chunks(self, chunks: List[DocumentChunk]):
        """Store document chunks with embeddings in Redis"""
        logger.info(f"Storing {len(chunks)} chunks in Redis")
        
        if not chunks:
            return
        
        # Create index if needed
        if chunks[0].embedding is not None:
            self.create_index(len(chunks[0].embedding))
        
        pipe = self.redis_client.pipeline()
        
        for chunk in chunks:
            if chunk.embedding is None:
                continue
                
            key = f"{self.index_name}:{chunk.id}"
            
            # Convert embedding to bytes
            embedding_bytes = chunk.embedding.astype(np.float32).tobytes()
            
            pipe.hset(key, mapping={
                "text": chunk.text,
                "source": chunk.source,
                "chunk_id": chunk.id,
                "embedding": embedding_bytes
            })
        
        pipe.execute()
        logger.info(f"✅ Stored {len(chunks)} chunks in Redis")
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search for query embedding"""
        from redis.commands.search.query import Query
        
        # Convert query embedding to bytes
        query_bytes = query_embedding.astype(np.float32).tobytes()
        
        # Create search query
        query = Query(f"*=>[KNN {k} @embedding $query_vector AS score]").sort_by("score").return_fields("text", "source", "chunk_id", "score").dialect(2)
        
        try:
            results = self.redis_client.ft(self.index_name).search(query, query_params={"query_vector": query_bytes})
            
            search_results = []
            for doc in results.docs:
                search_results.append({
                    "text": doc.text,
                    "source": doc.source,
                    "chunk_id": doc.chunk_id,
                    "score": float(doc.score)
                })
            
            logger.info(f"Found {len(search_results)} similar chunks")
            return search_results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

class RAGSystem:
    """Main RAG system orchestrating all components"""
    
    def __init__(self, redis_url: str, openai_api_key: str):
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = RedisVectorStore(redis_url)
        
        # Setup OpenAI
        openai.api_key = openai_api_key
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        logger.info("✅ RAG System initialized successfully")
    
    def process_document(self, source: str, doc_type: str) -> int:
        """Process a document and store it in the vector database"""
        logger.info(f"Processing document: {source} (type: {doc_type})")
        
        # Process document based on type
        if doc_type == "1":  # Wikipedia URL
            chunks = self.document_processor.process_wikipedia_url(source)
        elif doc_type == "2":  # Text file
            chunks = self.document_processor.process_text_file(source)
        elif doc_type == "3":  # PDF file
            chunks = self.document_processor.process_pdf_file(source)
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")
        
        # Generate embeddings
        chunks_with_embeddings = self.embedding_generator.generate_embeddings(chunks)
        
        # Store in vector database
        self.vector_store.store_chunks(chunks_with_embeddings)
        
        logger.info(f"✅ Successfully processed and stored {len(chunks)} chunks")
        return len(chunks)
    
    def query(self, question: str, k: int = 5) -> str:
        """Answer a question using RAG"""
        logger.info(f"Processing query: {question}")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(question)
        
        # Retrieve relevant chunks
        relevant_chunks = self.vector_store.similarity_search(query_embedding, k=k)
        
        if not relevant_chunks:
            return "I couldn't find any relevant information to answer your question."
        
        # Prepare context for OpenAI
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
        
        # Generate response using OpenAI
        prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            # Log retrieval info
            logger.info(f"Retrieved {len(relevant_chunks)} chunks for context")
            for i, chunk in enumerate(relevant_chunks):
                logger.info(f"  Chunk {i+1}: Score {chunk['score']:.4f} from {chunk['source']}")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error while generating the response: {e}"

def main():
    """Main application entry point"""
    print("🚀 RAG Workshop - Interactive Demo")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    redis_url = os.getenv('REDIS_URL')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    document_source = os.getenv('DOCUMENT_SOURCE')
    document_type = os.getenv('DOCUMENT_TYPE')
    
    if not all([redis_url, openai_api_key, document_source, document_type]):
        print("❌ Missing configuration. Please run setup.py first.")
        sys.exit(1)
    
    try:
        # Initialize RAG system
        rag = RAGSystem(redis_url, openai_api_key)
        
        # Process document
        print(f"\n📄 Processing document: {document_source}")
        num_chunks = rag.process_document(document_source, document_type)
        print(f"✅ Document processed into {num_chunks} chunks")
        
        # Interactive query loop
        print(f"\n💬 Ready for questions! (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            question = input("\n🤔 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            print("\n🔍 Searching for relevant information...")
            answer = rag.query(question)
            print(f"\n💡 Answer:\n{answer}")
            print("-" * 50)
        
        print("\n👋 Thanks for using the RAG Workshop demo!")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
