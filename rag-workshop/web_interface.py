#!/usr/bin/env python3
"""
RAG Workshop - Streamlit Web Interface
A user-friendly web interface for the RAG system demonstration.
"""

import os
import streamlit as st
from dotenv import load_dotenv
import logging
from datetime import datetime
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
            st.error("Missing configuration. Please run setup.py first.")
            st.stop()

        try:
            st.session_state.rag_system = RAGSystem(redis_url, openai_api_key)
            st.session_state.document_processed = False
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {e}")
            st.stop()

def process_document(force_reprocess=False):
    """Process the configured document"""
    if not st.session_state.document_processed or force_reprocess:
        document_source = os.getenv('DOCUMENT_SOURCE')
        document_type = os.getenv('DOCUMENT_TYPE')

        if not document_source or not document_type:
            st.error("Document configuration missing. Please run setup.py first.")
            return

        with st.spinner("Processing document..."):
            try:
                num_chunks = st.session_state.rag_system.process_document(
                    document_source, document_type, force_reprocess=force_reprocess
                )
                st.session_state.document_processed = True
                st.session_state.num_chunks = num_chunks
                st.session_state.document_source = document_source

                if force_reprocess:
                    st.success(f"Document reprocessed into {num_chunks} chunks!")
                else:
                    st.success(f"Document processed into {num_chunks} chunks!")
            except Exception as e:
                st.error(f"Error processing document: {e}")

def load_chat_history():
    """Load chat history from Redis"""
    if 'chat_history_loaded' not in st.session_state:
        document_source = os.getenv('DOCUMENT_SOURCE')
        if document_source and hasattr(st.session_state, 'rag_system'):
            try:
                redis_history = st.session_state.rag_system.get_chat_history(document_source)
                st.session_state.chat_history = redis_history
                st.session_state.chat_history_loaded = True
                logger.info(f"Loaded {len(redis_history)} chat entries from Redis")
            except Exception as e:
                logger.error(f"Error loading chat history: {e}")
                st.session_state.chat_history = []
                st.session_state.chat_history_loaded = True

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="RAG Workshop Demo",
        page_icon="⚡",
        layout="wide"
    )

    st.title("RAG Workshop - Interactive Demo")
    st.markdown("---")
    
    # Initialize RAG system
    initialize_rag_system()
    
    # Sidebar with configuration info
    with st.sidebar:
        st.header("Configuration")
        
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

        # Document management buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reprocess Document"):
                process_document(force_reprocess=True)
                st.rerun()

        with col2:
            if st.button("Clear Embeddings"):
                if st.session_state.get('rag_system'):
                    st.session_state.rag_system.clear_document_embeddings(document_source)
                    st.session_state.document_processed = False
                    st.success("Embeddings cleared!")
                    st.rerun()

        # Show indexed documents
        st.markdown("---")
        st.subheader("Indexed Documents")

        if hasattr(st.session_state, 'rag_system'):
            try:
                indexed_docs = st.session_state.rag_system.get_indexed_documents()
                if indexed_docs:
                    for doc in indexed_docs:
                        with st.expander(f"📄 {doc.get('source', 'Unknown')[:50]}..."):
                            st.text(f"Type: {doc_type_map.get(doc.get('type', ''), 'Unknown')}")
                            st.text(f"Chunks: {doc.get('chunk_count', 'Unknown')}")
                            st.text(f"Processed: {doc.get('processed_date', 'Unknown')[:19]}")

                            if st.button(f"Clear This Document", key=f"clear_{doc.get('doc_hash', '')}"):
                                st.session_state.rag_system.clear_document_embeddings(doc.get('source'))
                                st.success(f"Cleared embeddings for {doc.get('source', 'document')}")
                                st.rerun()
                else:
                    st.text("No documents indexed")
            except Exception as e:
                st.error(f"Error loading indexed documents: {e}")

        # Semantic Cache Management
        st.markdown("---")
        st.subheader("Semantic Cache")

        # Semantic cache toggle
        use_cache = st.checkbox("Enable Semantic Cache",
                               value=st.session_state.get('use_semantic_cache', False),
                               help="Cache similar questions for faster responses")
        st.session_state.use_semantic_cache = use_cache

        # Cache threshold slider
        cache_threshold = st.slider("Similarity Threshold",
                                   min_value=0.7, max_value=0.95,
                                   value=st.session_state.get('cache_threshold', 0.85),
                                   step=0.05,
                                   help="Higher = more strict matching")
        st.session_state.cache_threshold = cache_threshold

        # Cache statistics
        if hasattr(st.session_state, 'rag_system'):
            try:
                cache_stats = st.session_state.rag_system.get_cache_stats(document_source)
                st.metric("Cache Entries", cache_stats.get('document_cache_entries', 0))

                # Clear cache button
                if st.button("Clear Semantic Cache"):
                    st.session_state.rag_system.clear_semantic_cache(document_source)
                    st.success("Semantic cache cleared!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading cache stats: {e}")

        # Chat history management
        st.markdown("---")
        st.subheader("Chat History")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat History"):
                if hasattr(st.session_state, 'rag_system'):
                    st.session_state.rag_system.clear_chat_history(document_source)
                    st.session_state.chat_history = []
                    st.session_state.chat_history_loaded = False
                    st.success("Chat history cleared!")
                    st.rerun()

        with col2:
            if st.button("Reload History"):
                st.session_state.chat_history_loaded = False
                load_chat_history()
                st.success("Chat history reloaded!")
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Ask Questions")

        # Process document if not already done
        if not st.session_state.get('document_processed', False):
            process_document()

        # Load chat history from Redis
        load_chat_history()

        if st.session_state.get('document_processed', False):
            st.success(f"Document ready! ({st.session_state.get('num_chunks', 0)} chunks indexed)")

            # Query interface
            question = st.text_input(
                "Your question:",
                placeholder="Ask anything about the document...",
                key="question_input"
            )

            col_ask, col_clear = st.columns([1, 4])
            with col_ask:
                ask_button = st.button("Ask", type="primary")
            with col_clear:
                if st.button("Clear History"):
                    document_source = os.getenv('DOCUMENT_SOURCE')
                    if hasattr(st.session_state, 'rag_system'):
                        st.session_state.rag_system.clear_chat_history(document_source)
                    st.session_state.chat_history = []
                    st.session_state.chat_history_loaded = False
                    st.rerun()

            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            # Process question
            if ask_button and question:
                # Get cache settings
                use_cache = st.session_state.get('use_semantic_cache', False)
                cache_threshold = st.session_state.get('cache_threshold', 0.85)

                spinner_text = "Checking semantic cache..." if use_cache else "Searching for relevant information..."

                with st.spinner(spinner_text):
                    try:
                        document_source = os.getenv('DOCUMENT_SOURCE')
                        answer, metadata = st.session_state.rag_system.query(
                            question,
                            document_source=document_source,
                            store_history=True,
                            use_semantic_cache=use_cache,
                            cache_threshold=cache_threshold
                        )

                        # Show performance metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if metadata['cache_hit']:
                                st.success(f"🚀 Cache Hit! ({metadata['response_time_ms']}ms)")
                            else:
                                st.info(f"⏱️ Full Search ({metadata['response_time_ms']}ms)")

                        with col2:
                            if metadata['cache_hit']:
                                st.metric("Similarity", f"{metadata['similarity_score']:.3f}")
                            else:
                                st.metric("Chunks", metadata['chunks_retrieved'])

                        with col3:
                            if use_cache:
                                cache_stats = st.session_state.rag_system.get_cache_stats(document_source)
                                st.metric("Cache Size", cache_stats.get('document_cache_entries', 0))

                        # Add to local session state for immediate display
                        new_entry = {
                            'question': question,
                            'answer': answer,
                            'timestamp': datetime.now().isoformat(),
                            'document_source': document_source,
                            'cache_hit': metadata['cache_hit'],
                            'response_time_ms': metadata['response_time_ms'],
                            'similarity_score': metadata.get('similarity_score', 0.0)
                        }
                        st.session_state.chat_history.insert(0, new_entry)

                        # Rerun to refresh the page (input will be cleared automatically)
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error processing question: {e}")

            # Display chat history
            if st.session_state.chat_history:
                st.markdown("---")
                st.header("Conversation History")

                for i, chat in enumerate(st.session_state.chat_history):
                    # Format timestamp
                    timestamp = chat.get('timestamp', '')
                    if timestamp:
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            time_str = dt.strftime('%Y-%m-%d %H:%M')
                        except:
                            time_str = timestamp[:19]
                    else:
                        time_str = 'Unknown time'

                    # Add cache indicator to title
                    cache_indicator = ""
                    if chat.get('cache_hit'):
                        cache_indicator = " 🚀"
                    elif 'response_time_ms' in chat:
                        cache_indicator = " ⏱️"

                    with st.expander(f"Q{i+1}: {chat['question'][:50]}...{cache_indicator} ({time_str})", expanded=(i==0)):
                        st.markdown(f"**Question:** {chat['question']}")
                        st.markdown(f"**Answer:** {chat['answer']}")

                        # Performance info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.caption(f"📄 Document: {chat.get('document_source', 'Unknown')}")
                        with col2:
                            if 'response_time_ms' in chat:
                                st.caption(f"⏱️ Response: {chat['response_time_ms']}ms")
                        with col3:
                            if chat.get('cache_hit'):
                                st.caption(f"🚀 Cache Hit (sim: {chat.get('similarity_score', 0):.3f})")
                            else:
                                st.caption(f"🔍 Full Search")

                        st.caption(f"🕒 Time: {time_str}")
    
    with col2:
        st.header("How RAG Works")

        with st.expander("1. Document Processing", expanded=True):
            st.markdown("""
            - **Extract text** from various sources (Wikipedia, PDFs, text files)
            - **Split into chunks** for better processing
            - **Clean and preprocess** the text
            """)
        
        with st.expander("2. Embedding Generation"):
            st.markdown("""
            - **Convert text to vectors** using sentence transformers
            - **Create semantic representations** of the content
            - **Enable similarity comparisons** between text pieces
            """)

        with st.expander("3. Vector Storage"):
            st.markdown("""
            - **Store embeddings** in Redis vector database
            - **Create search index** for fast retrieval
            - **Enable similarity search** across all chunks
            """)

        with st.expander("4. Retrieval"):
            st.markdown("""
            - **Convert question** to embedding
            - **Find similar chunks** using cosine similarity
            - **Rank results** by relevance score
            """)

        with st.expander("5. Generation"):
            st.markdown("""
            - **Provide context** to OpenAI GPT model
            - **Generate answer** based on retrieved information
            - **Return contextual response** to user
            """)
        
        # System status
        st.markdown("---")
        st.header("System Status")

        # Check Redis connection
        try:
            st.session_state.rag_system.vector_store.redis_client.ping()
            st.success("Redis Connected")
        except:
            st.error("Redis Disconnected")

        # Document status
        if st.session_state.get('document_processed', False):
            st.success(f"Document Indexed ({st.session_state.get('num_chunks', 0)} chunks)")
        else:
            st.warning("Document Not Processed")

        # OpenAI status
        if os.getenv('OPENAI_API_KEY'):
            st.success("OpenAI Configured")
        else:
            st.error("OpenAI Not Configured")

if __name__ == "__main__":
    main()
