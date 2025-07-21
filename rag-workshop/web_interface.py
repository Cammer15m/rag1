#!/usr/bin/env python3
"""
RAG Workshop - Streamlit Web Interface
A user-friendly web interface for the RAG system demonstration.
"""

import os
import streamlit as st
from dotenv import load_dotenv
import logging
from main import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def initialize_rag_system():
    """Initialize the RAG system with caching"""
    if 'rag_system' not in st.session_state:
        redis_url = os.getenv('REDIS_URL')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not redis_url or not openai_api_key:
            st.error("❌ Missing configuration. Please run setup.py first.")
            st.stop()
        
        try:
            st.session_state.rag_system = RAGSystem(redis_url, openai_api_key)
            st.session_state.document_processed = False
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {e}")
            st.stop()

def process_document():
    """Process the configured document"""
    if not st.session_state.document_processed:
        document_source = os.getenv('DOCUMENT_SOURCE')
        document_type = os.getenv('DOCUMENT_TYPE')
        
        if not document_source or not document_type:
            st.error("❌ Document configuration missing. Please run setup.py first.")
            return
        
        with st.spinner("📄 Processing document..."):
            try:
                num_chunks = st.session_state.rag_system.process_document(document_source, document_type)
                st.session_state.document_processed = True
                st.session_state.num_chunks = num_chunks
                st.success(f"✅ Document processed into {num_chunks} chunks!")
            except Exception as e:
                st.error(f"❌ Error processing document: {e}")

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="RAG Workshop Demo",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 RAG Workshop - Interactive Demo")
    st.markdown("---")
    
    # Initialize RAG system
    initialize_rag_system()
    
    # Sidebar with configuration info
    with st.sidebar:
        st.header("📋 Configuration")
        
        redis_url = os.getenv('REDIS_URL', 'Not configured')
        document_source = os.getenv('DOCUMENT_SOURCE', 'Not configured')
        document_type = os.getenv('DOCUMENT_TYPE', 'Not configured')
        
        # Mask sensitive information
        if redis_url.startswith('redis://'):
            if '@' in redis_url:
                # Cloud Redis - mask credentials
                parts = redis_url.split('@')
                masked_url = f"{parts[0].split('//')[0]}//***:***@{parts[1]}"
            else:
                # Local Redis
                masked_url = redis_url
        else:
            masked_url = redis_url
        
        st.text(f"Redis: {masked_url}")
        st.text(f"Document: {document_source}")
        
        doc_type_map = {"1": "Wikipedia URL", "2": "Text File", "3": "PDF File"}
        st.text(f"Type: {doc_type_map.get(document_type, 'Unknown')}")
        
        if st.button("🔄 Reprocess Document"):
            st.session_state.document_processed = False
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Process document if not already done
        if not st.session_state.get('document_processed', False):
            process_document()
        
        if st.session_state.get('document_processed', False):
            st.success(f"📚 Document ready! ({st.session_state.get('num_chunks', 0)} chunks indexed)")
            
            # Query interface
            question = st.text_input(
                "🤔 Your question:",
                placeholder="Ask anything about the document...",
                key="question_input"
            )
            
            col_ask, col_clear = st.columns([1, 4])
            with col_ask:
                ask_button = st.button("🔍 Ask", type="primary")
            with col_clear:
                if st.button("🗑️ Clear History"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Process question
            if ask_button and question:
                with st.spinner("🔍 Searching for relevant information..."):
                    try:
                        answer = st.session_state.rag_system.query(question)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'question': question,
                            'answer': answer
                        })
                        
                        # Clear input
                        st.session_state.question_input = ""
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error processing question: {e}")
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("---")
                st.header("💭 Conversation History")
                
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:50]}...", expanded=(i==0)):
                        st.markdown(f"**🤔 Question:** {chat['question']}")
                        st.markdown(f"**💡 Answer:** {chat['answer']}")
    
    with col2:
        st.header("📖 How RAG Works")
        
        with st.expander("1. 📄 Document Processing", expanded=True):
            st.markdown("""
            - **Extract text** from various sources (Wikipedia, PDFs, text files)
            - **Split into chunks** for better processing
            - **Clean and preprocess** the text
            """)
        
        with st.expander("2. 🧠 Embedding Generation"):
            st.markdown("""
            - **Convert text to vectors** using sentence transformers
            - **Create semantic representations** of the content
            - **Enable similarity comparisons** between text pieces
            """)
        
        with st.expander("3. 🗄️ Vector Storage"):
            st.markdown("""
            - **Store embeddings** in Redis vector database
            - **Create search index** for fast retrieval
            - **Enable similarity search** across all chunks
            """)
        
        with st.expander("4. 🔍 Retrieval"):
            st.markdown("""
            - **Convert question** to embedding
            - **Find similar chunks** using cosine similarity
            - **Rank results** by relevance score
            """)
        
        with st.expander("5. 🤖 Generation"):
            st.markdown("""
            - **Provide context** to OpenAI GPT model
            - **Generate answer** based on retrieved information
            - **Return contextual response** to user
            """)
        
        # System status
        st.markdown("---")
        st.header("🔧 System Status")
        
        # Check Redis connection
        try:
            st.session_state.rag_system.vector_store.redis_client.ping()
            st.success("✅ Redis Connected")
        except:
            st.error("❌ Redis Disconnected")
        
        # Document status
        if st.session_state.get('document_processed', False):
            st.success(f"✅ Document Indexed ({st.session_state.get('num_chunks', 0)} chunks)")
        else:
            st.warning("⏳ Document Not Processed")
        
        # OpenAI status
        if os.getenv('OPENAI_API_KEY'):
            st.success("✅ OpenAI Configured")
        else:
            st.error("❌ OpenAI Not Configured")

if __name__ == "__main__":
    main()
