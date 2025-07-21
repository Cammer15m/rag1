#!/usr/bin/env python3
"""
RAG Workshop - Testing and Validation Script
Tests the RAG system with sample queries to ensure everything works correctly.
"""

import os
import sys
import time
from dotenv import load_dotenv
from main import RAGSystem

def test_rag_system():
    """Test the RAG system with sample queries"""
    print("🧪 RAG System Testing")
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
        return False
    
    try:
        # Initialize RAG system
        print("🚀 Initializing RAG system...")
        rag = RAGSystem(redis_url, openai_api_key)
        print("✅ RAG system initialized")
        
        # Process document
        print(f"\n📄 Processing document: {document_source}")
        start_time = time.time()
        num_chunks = rag.process_document(document_source, document_type)
        processing_time = time.time() - start_time
        print(f"✅ Document processed into {num_chunks} chunks in {processing_time:.2f} seconds")
        
        # Test queries
        test_queries = [
            "What is artificial intelligence?",
            "What are the types of AI?",
            "What is machine learning?",
            "What are the applications of AI?",
            "What are the challenges of AI?",
            "When was AI research founded?",
            "What is deep learning?",
            "What is the future of AI?"
        ]
        
        print(f"\n🔍 Testing with {len(test_queries)} sample queries...")
        print("-" * 50)
        
        successful_queries = 0
        total_query_time = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Testing: {query}")
            
            try:
                start_time = time.time()
                answer = rag.query(query)
                query_time = time.time() - start_time
                total_query_time += query_time
                
                if answer and len(answer) > 50:  # Basic validation
                    print(f"   ✅ Response received ({len(answer)} chars, {query_time:.2f}s)")
                    print(f"   📝 Preview: {answer[:100]}...")
                    successful_queries += 1
                else:
                    print(f"   ❌ Poor response: {answer}")
                    
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Results Summary")
        print("=" * 50)
        print(f"✅ Successful queries: {successful_queries}/{len(test_queries)}")
        print(f"⏱️  Average query time: {total_query_time/len(test_queries):.2f} seconds")
        print(f"📄 Document chunks: {num_chunks}")
        print(f"⚡ Processing time: {processing_time:.2f} seconds")
        
        success_rate = successful_queries / len(test_queries)
        if success_rate >= 0.8:
            print("\n🎉 RAG system is working well!")
            return True
        elif success_rate >= 0.5:
            print("\n⚠️  RAG system has some issues but is functional")
            return True
        else:
            print("\n❌ RAG system has significant issues")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_components():
    """Test individual components"""
    print("\n🔧 Component Testing")
    print("-" * 30)
    
    try:
        from main import DocumentProcessor, EmbeddingGenerator, RedisVectorStore
        
        # Test document processor
        print("1. Testing DocumentProcessor...")
        processor = DocumentProcessor()
        
        # Test with sample text
        sample_text = "This is a test document. It has multiple sentences. We will chunk this text."
        chunks = processor._chunk_text(sample_text, "test_source")
        print(f"   ✅ Created {len(chunks)} chunks from sample text")
        
        # Test embedding generator
        print("2. Testing EmbeddingGenerator...")
        embedder = EmbeddingGenerator()
        embeddings = embedder.generate_embeddings(chunks[:2])  # Test with first 2 chunks
        print(f"   ✅ Generated embeddings with dimension {embedder.embedding_dim}")
        
        # Test Redis connection
        print("3. Testing Redis connection...")
        redis_url = os.getenv('REDIS_URL')
        if redis_url:
            vector_store = RedisVectorStore(redis_url)
            print("   ✅ Redis connection successful")
        else:
            print("   ❌ Redis URL not configured")
        
        print("✅ All components tested successfully")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def interactive_test():
    """Interactive testing mode"""
    print("\n💬 Interactive Testing Mode")
    print("-" * 30)
    print("Enter questions to test the RAG system (type 'quit' to exit)")
    
    # Load environment and initialize RAG
    load_dotenv()
    redis_url = os.getenv('REDIS_URL')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not redis_url or not openai_api_key:
        print("❌ Missing configuration for interactive mode")
        return
    
    try:
        rag = RAGSystem(redis_url, openai_api_key)
        
        while True:
            question = input("\n🤔 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            try:
                start_time = time.time()
                answer = rag.query(question)
                query_time = time.time() - start_time
                
                print(f"\n💡 Answer ({query_time:.2f}s):")
                print(answer)
                print("-" * 50)
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("👋 Interactive testing completed")
        
    except Exception as e:
        print(f"❌ Failed to initialize interactive mode: {e}")

def main():
    """Main testing function"""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "components":
            test_components()
        elif mode == "interactive":
            interactive_test()
        elif mode == "full":
            success = test_components()
            if success:
                test_rag_system()
        else:
            print("Usage: python test_rag.py [components|interactive|full]")
            print("  components  - Test individual components")
            print("  interactive - Interactive question mode")
            print("  full        - Run all tests")
            print("  (no args)   - Run standard RAG test")
    else:
        # Default: run standard RAG test
        test_rag_system()

if __name__ == "__main__":
    main()
