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
import tempfile
import requests
from urllib.parse import urlparse
import PyPDF2
import io

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

def process_uploaded_file(uploaded_file, file_type):
    """Process uploaded file and return content"""
    try:
        if file_type == "text":
            # Read text file
            content = uploaded_file.read().decode('utf-8')
            return content, f"uploaded_{uploaded_file.name}"

        elif file_type == "pdf":
            # Read PDF file
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
            return content, f"uploaded_{uploaded_file.name}"

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None

def process_url(url):
    """Process URL and return content"""
    try:
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            st.error("Please enter a valid URL (including http:// or https://)")
            return None, None

        # Fetch content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # For now, just return the raw HTML - in a real implementation you'd parse it
        content = response.text
        return content, url

    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return None, None

def upload_and_process_document(content, source, doc_type):
    """Upload and process a new document"""
    try:
        # Process the document with content
        num_chunks = st.session_state.rag_system.process_document(
            source, doc_type, force_reprocess=True, content=content
        )

        # Update session state
        st.session_state.document_processed = True
        st.session_state.num_chunks = num_chunks
        st.session_state.document_source = source
        st.session_state.chat_history = []  # Clear chat history for new document
        st.session_state.chat_history_loaded = False

        # Clear semantic cache since we have a new document
        st.session_state.rag_system.clear_semantic_cache()

        return num_chunks

    except Exception as e:
        st.error(f"Error processing document: {e}")
        return None

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

    # Document Upload Section
    st.header("📁 Document Upload")
    st.markdown("Upload a new document to demonstrate RAG with Redis caching")

    upload_col1, upload_col2 = st.columns([2, 1])

    with upload_col1:
        upload_method = st.radio(
            "Choose upload method:",
            ["File Upload", "URL"],
            horizontal=True
        )

        if upload_method == "File Upload":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'pdf'],
                help="Upload a text file (.txt) or PDF file (.pdf)"
            )

            if uploaded_file is not None:
                file_type = "text" if uploaded_file.name.endswith('.txt') else "pdf"

                if st.button("Process Uploaded File", type="primary"):
                    with st.spinner("Processing uploaded file..."):
                        content, source = process_uploaded_file(uploaded_file, file_type)
                        if content and source:
                            num_chunks = upload_and_process_document(content, source, "2" if file_type == "text" else "3")
                            if num_chunks:
                                st.success(f"✅ File processed successfully! {num_chunks} chunks indexed.")
                                st.rerun()

        else:  # URL
            url_input = st.text_input(
                "Enter URL:",
                placeholder="https://example.com/article",
                help="Enter a URL to fetch and process content"
            )

            if url_input and st.button("Process URL", type="primary"):
                with st.spinner("Fetching and processing URL..."):
                    content, source = process_url(url_input)
                    if content and source:
                        num_chunks = upload_and_process_document(content, source, "1")
                        if num_chunks:
                            st.success(f"✅ URL processed successfully! {num_chunks} chunks indexed.")
                            st.rerun()

    with upload_col2:
        st.info("""
        **💡 Cache Demo Tips:**

        1. Upload a document
        2. Ask a question (slow - no cache)
        3. Ask the same question again (fast - cache hit!)
        4. Upload a different document
        5. Ask the same question (slow again - different context)

        This demonstrates how Redis caching speeds up repeated queries while maintaining context accuracy.
        """)

    st.markdown("---")
    
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
                        st.markdown(f"📄 **{doc.get('source', 'Unknown')[:50]}...**")
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

        # Semantic cache implementation selector
        cache_options = ["Disabled", "Custom Implementation", "LangChain Implementation"]
        cache_impl = st.radio(
            "Cache Implementation:",
            cache_options,
            index=st.session_state.get('cache_implementation_index', 1),  # Default to Custom
            help="Choose semantic cache implementation to test different approaches"
        )

        # Store the selection
        cache_impl_index = cache_options.index(cache_impl)
        st.session_state.cache_implementation_index = cache_impl_index

        # Set use_semantic_cache based on selection
        use_cache = cache_impl != "Disabled"
        st.session_state.use_semantic_cache = use_cache
        st.session_state.cache_implementation = cache_impl

        # Show implementation details
        if cache_impl == "Custom Implementation":
            st.success("✅ Using our custom semantic cache (working)")
        elif cache_impl == "LangChain Implementation":
            st.warning("⚠️ Using LangChain Redis cache (may have compatibility issues)")
        else:
            st.info("ℹ️ Semantic cache disabled - all queries will be full searches")

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

            # Query interface with history dropdown
            col_input, col_history = st.columns([3, 1])

            with col_input:
                # Handle pending question from history selection
                default_value = ""
                if 'pending_question' in st.session_state:
                    default_value = st.session_state.pending_question
                    del st.session_state.pending_question

                question = st.text_input(
                    "Your question:",
                    value=default_value,
                    placeholder="Ask anything about the document...",
                    key="question_input"
                )

            with col_history:
                # History dropdown - only show if there's history
                if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
                    history_options = ["Select from history..."] + [
                        f"Q{i+1}: {chat['question'][:50]}..." if len(chat['question']) > 50
                        else f"Q{i+1}: {chat['question']}"
                        for i, chat in enumerate(st.session_state.chat_history)
                    ]
                    selected_history = st.selectbox(
                        "Previous Questions:",
                        history_options,
                        key="history_selector"
                    )

                    if selected_history != "Select from history...":
                        # Extract the index and set the question
                        history_index = history_options.index(selected_history) - 1
                        selected_question = st.session_state.chat_history[history_index]['question']
                        # Use a different approach - set a flag to update the input
                        st.session_state.pending_question = selected_question
                        st.rerun()
                else:
                    # Show placeholder when no history
                    st.write("")  # Empty space to maintain layout

            # Ask button, loop option, and clear history
            col_ask, col_loop, col_clear = st.columns([1, 1, 2])

            with col_ask:
                ask_button = st.button("Ask", type="primary")

            with col_loop:
                loop_mode = st.checkbox("Loop Mode", help="Automatically ask the same question multiple times for performance testing")
                if loop_mode:
                    loop_count = st.number_input("Count:", min_value=1, max_value=10, value=3, key="loop_count")

            with col_clear:
                if st.button("Clear History"):
                    document_source = os.getenv('DOCUMENT_SOURCE')
                    if hasattr(st.session_state, 'rag_system'):
                        st.session_state.rag_system.clear_chat_history(document_source)
                    st.session_state.chat_history = []
                    st.session_state.chat_history_loaded = False

                    # Clear any pending question state
                    if 'pending_question' in st.session_state:
                        del st.session_state.pending_question

                    st.rerun()

            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            # Process question
            if ask_button and question:
                # Get cache settings
                use_cache = st.session_state.get('use_semantic_cache', False)
                cache_threshold = st.session_state.get('cache_threshold', 0.85)

                # Determine number of iterations
                iterations = loop_count if loop_mode else 1

                # Store results for loop mode analysis
                loop_results = []

                for iteration in range(iterations):
                    if iterations > 1:
                        st.markdown(f"### 🔄 Iteration {iteration + 1} of {iterations}")

                    spinner_text = "Checking semantic cache..." if use_cache else "Searching for relevant information..."
                    if iterations > 1:
                        spinner_text += f" (Iteration {iteration + 1}/{iterations})"

                    with st.spinner(spinner_text):
                        try:
                            document_source = os.getenv('DOCUMENT_SOURCE')
                            cache_implementation = st.session_state.get('cache_implementation', 'Custom Implementation')
                            answer, metadata = st.session_state.rag_system.query(
                                question,
                                document_source=document_source,
                                store_history=(iteration == 0),  # Only store history for first iteration
                                use_semantic_cache=use_cache,
                                cache_threshold=cache_threshold,
                                cache_implementation=cache_implementation
                            )

                            # Store results for analysis
                            loop_results.append({
                                'iteration': iteration + 1,
                                'answer': answer,
                                'metadata': metadata
                            })

                            # Show performance metrics for this iteration
                            perf_col1, perf_col2, perf_col3 = st.columns(3)
                            with perf_col1:
                                if metadata['cache_hit']:
                                    st.success(f"🚀 Cache Hit! ({metadata['response_time_ms']}ms)")
                                else:
                                    st.info(f"⏱️ Full Search ({metadata['response_time_ms']}ms)")

                            with perf_col2:
                                if metadata['cache_hit']:
                                    st.metric("Similarity", f"{metadata['similarity_score']:.3f}")
                                else:
                                    st.metric("Chunks", metadata['chunks_retrieved'])

                            with perf_col3:
                                if use_cache:
                                    cache_stats = st.session_state.rag_system.get_cache_stats(document_source)
                                    st.metric("Cache Size", cache_stats.get('document_cache_entries', 0))

                            # Show answer for this iteration
                            if iterations > 1:
                                with st.expander(f"Answer - Iteration {iteration + 1}", expanded=(iteration == 0)):
                                    st.write(answer)
                            else:
                                st.markdown("### Answer")
                                st.write(answer)

                        except Exception as e:
                            st.error(f"Error processing question (Iteration {iteration + 1}): {e}")
                            break

                # Show loop performance analysis if multiple iterations
                if iterations > 1 and loop_results:
                    st.markdown("### 📊 Loop Performance Analysis")

                    # Calculate statistics
                    response_times = [r['metadata']['response_time_ms'] for r in loop_results]
                    cache_hits = [r['metadata']['cache_hit'] for r in loop_results]

                    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

                    with col_stats1:
                        st.metric("Avg Response Time", f"{sum(response_times)/len(response_times):.1f}ms")

                    with col_stats2:
                        st.metric("Min Response Time", f"{min(response_times)}ms")

                    with col_stats3:
                        st.metric("Max Response Time", f"{max(response_times)}ms")

                    with col_stats4:
                        cache_hit_rate = sum(cache_hits) / len(cache_hits) * 100
                        st.metric("Cache Hit Rate", f"{cache_hit_rate:.1f}%")

                    # Show detailed results table
                    with st.expander("Detailed Results"):
                        import pandas as pd
                        df_data = []
                        for r in loop_results:
                            df_data.append({
                                'Iteration': r['iteration'],
                                'Response Time (ms)': r['metadata']['response_time_ms'],
                                'Cache Hit': '✅' if r['metadata']['cache_hit'] else '❌',
                                'Similarity Score': f"{r['metadata'].get('similarity_score', 0):.3f}",
                                'Chunks Retrieved': r['metadata'].get('chunks_retrieved', 0)
                            })
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True)

                # Add to chat history (only once, using first iteration's result)
                if loop_results:
                    first_result = loop_results[0]
                    new_entry = {
                        'question': question,
                        'answer': first_result['answer'],
                        'timestamp': datetime.now().isoformat(),
                        'document_source': document_source,
                        'cache_hit': first_result['metadata']['cache_hit'],
                        'response_time_ms': first_result['metadata']['response_time_ms'],
                        'similarity_score': first_result['metadata'].get('similarity_score', 0.0)
                    }
                    st.session_state.chat_history.insert(0, new_entry)

                    # Clear any pending question state
                    if 'pending_question' in st.session_state:
                        del st.session_state.pending_question

                    # Rerun to refresh the page (input will be cleared automatically)
                    st.rerun()

            # Display selected conversation history
            if st.session_state.chat_history:
                st.markdown("---")

                # History selector for viewing
                col_title, col_selector = st.columns([2, 1])

                with col_title:
                    st.header("Conversation History")

                with col_selector:
                    view_options = ["Show Latest"] + [
                        f"Q{i+1}: {chat['question'][:30]}..." if len(chat['question']) > 30
                        else f"Q{i+1}: {chat['question']}"
                        for i, chat in enumerate(st.session_state.chat_history)
                    ]
                    selected_view = st.selectbox(
                        "View:",
                        view_options,
                        key="history_viewer"
                    )

                # Determine which conversation to show
                if selected_view == "Show Latest":
                    chat_to_show = st.session_state.chat_history[0]
                    chat_index = 0
                else:
                    # Extract index from selection
                    chat_index = view_options.index(selected_view) - 1
                    chat_to_show = st.session_state.chat_history[chat_index]

                # Display the selected conversation
                chat = chat_to_show
                i = chat_index

                # Format timestamp
                timestamp = chat.get('timestamp', '')
                if timestamp:
                    try:
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

                st.markdown(f"### Q{i+1}: {chat['question'][:50]}...{cache_indicator}")
                st.markdown(f"**Time:** {time_str}")
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")

                # Performance info
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.caption(f"📄 Document: {chat.get('document_source', 'Unknown')}")
                with info_col2:
                    if 'response_time_ms' in chat:
                        st.caption(f"⏱️ Response: {chat['response_time_ms']}ms")
                with info_col3:
                    if chat.get('cache_hit'):
                        st.caption(f"🚀 Cache Hit (sim: {chat.get('similarity_score', 0):.3f})")
                    else:
                        st.caption(f"🔍 Full Search")
    
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
