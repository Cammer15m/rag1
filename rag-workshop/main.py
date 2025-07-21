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
import hashlib
import uuid
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
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

# LangChain for semantic caching
try:
    from langchain_redis.cache import RedisSemanticCache
    from langchain_core.embeddings import Embeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain Redis not available. Semantic caching will be disabled.")

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

# LangChain Embeddings wrapper for our SentenceTransformer
class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper to make SentenceTransformer compatible with LangChain"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = self.model.encode(texts)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()

class RedisVectorStore:
    """Redis-based vector storage with similarity search"""
    
    def __init__(self, redis_url: str, index_name: str = "documents"):
        self.redis_client = redis.from_url(redis_url)
        self.index_name = index_name
        self.embedding_dim = None
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
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
        logger.info(f"Created Redis index '{self.index_name}' with dimension {embedding_dim}")
    
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
        logger.info(f"Stored {len(chunks)} chunks in Redis")
    
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

    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """Get list of documents currently indexed in Redis"""
        try:
            # Get all document metadata keys
            doc_keys = self.redis_client.keys("doc_meta:*")
            documents = []

            for key in doc_keys:
                doc_info = self.redis_client.hgetall(key)
                if doc_info:
                    # Convert bytes to strings
                    doc_info = {k.decode() if isinstance(k, bytes) else k:
                               v.decode() if isinstance(v, bytes) else v
                               for k, v in doc_info.items()}
                    documents.append(doc_info)

            return documents
        except Exception as e:
            logger.error(f"Error getting indexed documents: {e}")
            return []

    def store_document_metadata(self, source: str, doc_type: str, chunk_count: int):
        """Store metadata about processed document"""
        doc_hash = hashlib.md5(source.encode()).hexdigest()
        key = f"doc_meta:{doc_hash}"

        metadata = {
            "source": source,
            "type": doc_type,
            "chunk_count": chunk_count,
            "processed_date": datetime.now().isoformat(),
            "doc_hash": doc_hash
        }

        self.redis_client.hset(key, mapping=metadata)
        logger.info(f"Stored metadata for document: {source}")

    def clear_document_embeddings(self, source: str = None):
        """Clear embeddings for specific document or all documents"""
        try:
            if source:
                # Clear specific document
                doc_hash = hashlib.md5(source.encode()).hexdigest()

                # Get all chunk keys for this document
                chunk_keys = self.redis_client.keys(f"{self.index_name}:*")
                deleted_count = 0

                for key in chunk_keys:
                    chunk_data = self.redis_client.hget(key, "source")
                    if chunk_data and chunk_data.decode() == source:
                        self.redis_client.delete(key)
                        deleted_count += 1

                # Delete document metadata
                self.redis_client.delete(f"doc_meta:{doc_hash}")
                logger.info(f"Cleared {deleted_count} chunks for document: {source}")

            else:
                # Clear all documents
                chunk_keys = self.redis_client.keys(f"{self.index_name}:*")
                meta_keys = self.redis_client.keys("doc_meta:*")

                if chunk_keys:
                    self.redis_client.delete(*chunk_keys)
                if meta_keys:
                    self.redis_client.delete(*meta_keys)

                logger.info(f"Cleared all embeddings: {len(chunk_keys)} chunks, {len(meta_keys)} documents")

        except Exception as e:
            logger.error(f"Error clearing embeddings: {e}")

    def document_exists(self, source: str) -> bool:
        """Check if document is already indexed"""
        doc_hash = hashlib.md5(source.encode()).hexdigest()
        return self.redis_client.exists(f"doc_meta:{doc_hash}")

    def store_chat_entry(self, question: str, answer: str, document_source: str):
        """Store chat entry in Redis"""
        chat_id = str(uuid.uuid4())
        key = f"chat:{chat_id}"

        entry = {
            "question": question,
            "answer": answer,
            "document_source": document_source,
            "timestamp": datetime.now().isoformat(),
            "chat_id": chat_id
        }

        self.redis_client.hset(key, mapping=entry)
        logger.info(f"Stored chat entry: {chat_id}")

    def get_chat_history(self, document_source: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chat history, optionally filtered by document source"""
        try:
            chat_keys = self.redis_client.keys("chat:*")
            chat_entries = []

            for key in chat_keys:
                entry = self.redis_client.hgetall(key)
                if entry:
                    # Convert bytes to strings
                    entry = {k.decode() if isinstance(k, bytes) else k:
                            v.decode() if isinstance(v, bytes) else v
                            for k, v in entry.items()}

                    # Filter by document source if specified
                    if document_source is None or entry.get("document_source") == document_source:
                        chat_entries.append(entry)

            # Sort by timestamp (newest first)
            chat_entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            return chat_entries[:limit]

        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []

    def clear_chat_history(self, document_source: str = None):
        """Clear chat history, optionally filtered by document source"""
        try:
            chat_keys = self.redis_client.keys("chat:*")
            deleted_count = 0

            for key in chat_keys:
                if document_source:
                    # Check if this entry is for the specified document
                    entry_source = self.redis_client.hget(key, "document_source")
                    if entry_source and entry_source.decode() == document_source:
                        self.redis_client.delete(key)
                        deleted_count += 1
                else:
                    # Delete all chat entries
                    self.redis_client.delete(key)
                    deleted_count += 1

            logger.info(f"Cleared {deleted_count} chat entries")

        except Exception as e:
            logger.error(f"Error clearing chat history: {e}")

    def get_cache_stats(self, document_source: str = None) -> Dict[str, Any]:
        """Get semantic cache statistics"""
        try:
            # For Redis built-in semantic cache, we'll count cache keys
            cache_keys = self.redis_client.keys("llm_cache:*")
            return {
                "total_cache_entries": len(cache_keys),
                "document_cache_entries": len(cache_keys),
                "document_source": document_source
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"total_cache_entries": 0, "document_cache_entries": 0}

    def clear_semantic_cache(self, document_source: str = None):
        """Clear semantic cache"""
        try:
            cache_keys = self.redis_client.keys("llm_cache:*")
            if cache_keys:
                self.redis_client.delete(*cache_keys)
                logger.info(f"Cleared {len(cache_keys)} semantic cache entries")
        except Exception as e:
            logger.error(f"Error clearing semantic cache: {e}")

class RAGSystem:
    """Main RAG system orchestrating all components"""

    def __init__(self, redis_url: str, openai_api_key: str):
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = RedisVectorStore(redis_url)

        # Setup OpenAI (using legacy API for compatibility)
        openai.api_key = openai_api_key
        self.openai_client = None  # Not needed for legacy API

        # Initialize semantic cache if LangChain is available
        self.semantic_cache = None
        if LANGCHAIN_AVAILABLE:
            try:
                embeddings = SentenceTransformerEmbeddings()
                self.semantic_cache = RedisSemanticCache(
                    redis_url=redis_url,
                    embedding=embeddings,
                    score_threshold=0.85
                )
                logger.info("Semantic cache initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic cache: {e}")
                self.semantic_cache = None

        logger.info("RAG System initialized successfully")
    
    def process_document(self, source: str, doc_type: str, force_reprocess: bool = False) -> int:
        """Process a document and store it in the vector database"""
        logger.info(f"Processing document: {source} (type: {doc_type})")

        # Check if document already exists
        if not force_reprocess and self.vector_store.document_exists(source):
            logger.info(f"Document already indexed: {source}")
            # Get existing chunk count from metadata
            doc_hash = hashlib.md5(source.encode()).hexdigest()
            metadata = self.vector_store.redis_client.hgetall(f"doc_meta:{doc_hash}")
            if metadata:
                chunk_count = int(metadata.get(b"chunk_count", 0))
                logger.info(f"Using existing {chunk_count} chunks")
                return chunk_count

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

        # Store document metadata
        self.vector_store.store_document_metadata(source, doc_type, len(chunks))

        logger.info(f"Successfully processed and stored {len(chunks)} chunks")
        return len(chunks)
    
    def query(self, question: str, document_source: str = None, k: int = 5,
             store_history: bool = True, use_semantic_cache: bool = False,
             cache_threshold: float = 0.85) -> Tuple[str, Dict[str, Any]]:
        """Answer a question using RAG with optional semantic caching"""
        start_time = time.time()
        logger.info(f"Processing query: {question}")

        # Initialize response metadata
        metadata = {
            "cache_hit": False,
            "similarity_score": 0.0,
            "response_time_ms": 0,
            "chunks_retrieved": 0,
            "cache_enabled": use_semantic_cache and self.semantic_cache is not None
        }

        # Check semantic cache if enabled and available
        if use_semantic_cache and self.semantic_cache:
            try:
                # Update cache threshold
                self.semantic_cache.score_threshold = cache_threshold

                # Create a simple prompt for cache lookup
                cache_key = f"question:{question}"
                cached_result = self.semantic_cache.lookup(cache_key, question)

                if cached_result:
                    # Cache hit!
                    end_time = time.time()
                    metadata.update({
                        "cache_hit": True,
                        "similarity_score": cache_threshold,  # Approximate since Redis doesn't return exact score
                        "response_time_ms": round((end_time - start_time) * 1000, 2)
                    })

                    answer = cached_result

                    # Still store in chat history if requested
                    if store_history and document_source:
                        self.vector_store.store_chat_entry(question, answer, document_source)

                    logger.info(f"Semantic cache hit: {metadata['response_time_ms']}ms")
                    return answer, metadata

            except Exception as e:
                logger.warning(f"Semantic cache lookup failed: {e}")
                # Continue with normal processing

        # Cache miss or cache disabled - proceed with full RAG pipeline
        # Retrieve relevant chunks
        relevant_chunks = self.vector_store.similarity_search(query_embedding, k=k)
        metadata["chunks_retrieved"] = len(relevant_chunks)

        if not relevant_chunks:
            answer = "I couldn't find any relevant information to answer your question."
            end_time = time.time()
            metadata["response_time_ms"] = round((end_time - start_time) * 1000, 2)

            if store_history and document_source:
                self.vector_store.store_chat_entry(question, answer, document_source)
            return answer, metadata

        # Prepare context for OpenAI
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])

        # Generate response using OpenAI
        prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {question}

Answer:"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )

            answer = response.choices[0].message.content
            end_time = time.time()
            metadata["response_time_ms"] = round((end_time - start_time) * 1000, 2)

            # Store in semantic cache if enabled and successful
            if use_semantic_cache and self.semantic_cache and answer and not answer.startswith("Sorry"):
                try:
                    cache_key = f"question:{question}"
                    self.semantic_cache.update(cache_key, question, answer)
                    logger.info("Stored result in semantic cache")
                except Exception as e:
                    logger.warning(f"Failed to store in semantic cache: {e}")

            # Store chat history in Redis
            if store_history and document_source:
                self.vector_store.store_chat_entry(question, answer, document_source)

            # Log retrieval info
            logger.info(f"Retrieved {len(relevant_chunks)} chunks for context in {metadata['response_time_ms']}ms")
            for i, chunk in enumerate(relevant_chunks):
                logger.info(f"  Chunk {i+1}: Score {chunk['score']:.4f} from {chunk['source']}")

            return answer, metadata

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_answer = f"Sorry, I encountered an error while generating the response: {e}"
            end_time = time.time()
            metadata["response_time_ms"] = round((end_time - start_time) * 1000, 2)

            if store_history and document_source:
                self.vector_store.store_chat_entry(question, error_answer, document_source)
            return error_answer, metadata

    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """Get list of documents currently indexed"""
        return self.vector_store.get_indexed_documents()

    def clear_document_embeddings(self, source: str = None):
        """Clear embeddings for specific document or all documents"""
        self.vector_store.clear_document_embeddings(source)

    def get_chat_history(self, document_source: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chat history, optionally filtered by document source"""
        return self.vector_store.get_chat_history(document_source, limit)

    def clear_chat_history(self, document_source: str = None):
        """Clear chat history, optionally filtered by document source"""
        self.vector_store.clear_chat_history(document_source)

    def document_exists(self, source: str) -> bool:
        """Check if document is already indexed"""
        return self.vector_store.document_exists(source)

    def clear_semantic_cache(self, document_source: str = None):
        """Clear semantic cache"""
        if self.semantic_cache:
            try:
                self.semantic_cache.clear()
                logger.info("Semantic cache cleared")
            except Exception as e:
                logger.error(f"Failed to clear semantic cache: {e}")
        else:
            # Fallback to manual clearing
            self.vector_store.clear_semantic_cache(document_source)

    def get_cache_stats(self, document_source: str = None) -> Dict[str, Any]:
        """Get semantic cache statistics"""
        if self.semantic_cache:
            # For Redis semantic cache, we'll estimate based on Redis keys
            return self.vector_store.get_cache_stats(document_source)
        else:
            return {"total_cache_entries": 0, "document_cache_entries": 0}

def main():
    """Main application entry point"""
    print("RAG Workshop - Interactive Demo")
    print("=" * 50)

    # Load environment variables
    load_dotenv()

    # Get configuration
    redis_url = os.getenv('REDIS_URL')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    document_source = os.getenv('DOCUMENT_SOURCE')
    document_type = os.getenv('DOCUMENT_TYPE')

    if not all([redis_url, openai_api_key, document_source, document_type]):
        print("Missing configuration. Please run setup.py first.")
        sys.exit(1)
    
    try:
        # Initialize RAG system
        rag = RAGSystem(redis_url, openai_api_key)
        
        # Process document
        print(f"\nProcessing document: {document_source}")
        num_chunks = rag.process_document(document_source, document_type)
        print(f"Document processed into {num_chunks} chunks")

        # Interactive query loop
        print(f"\nReady for questions! (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            question = input("\n🤔 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            print("\nSearching for relevant information...")
            answer, metadata = rag.query(question, document_source=document_source)
            print(f"\nAnswer:\n{answer}")
            print(f"Response time: {metadata['response_time_ms']}ms")
            print("-" * 50)

        print("\nThanks for using the RAG Workshop demo!")

    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
