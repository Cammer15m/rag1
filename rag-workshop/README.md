# 🚀 RAG Workshop - Retrieval-Augmented Generation with Redis

A comprehensive workshop demonstrating how to build a Retrieval-Augmented Generation (RAG) system using Redis as a vector database and OpenAI for text generation.

## 🎯 Workshop Objectives

By the end of this workshop, participants will understand:

1. **Document Processing** - How to extract and chunk text from various sources
2. **Embeddings** - Converting text to vectors for semantic search
3. **Vector Databases** - Using Redis for storing and searching embeddings
4. **Retrieval** - Finding relevant context for user queries
5. **Generation** - Using LLMs to generate contextual responses

## 🏗️ Architecture Overview

```
📄 Document → 🔪 Chunking → 🧠 Embeddings → 🗄️ Redis Vector DB
                                                      ↓
🤖 OpenAI ← 📝 Context ← 🔍 Similarity Search ← 💬 User Query
```

## 🛠️ Prerequisites

- **Docker Desktop** installed and running
- **OpenAI API key**
- **That's it!** No Python installation required!

### 🍎 Mac Users
**Important**: Docker Desktop must be properly installed and running. If you're new to Docker on Mac, see [MAC_SETUP.md](MAC_SETUP.md) for detailed installation instructions.

**Quick test**: Run `docker --version` in Terminal. If it works, you're ready!

## 🚀 Quick Start (1 command!)

### Super Simple Workshop Start
```bash
# This handles everything - setup, presentation, and launch!
./workshop.sh
```

**What this does:**
- ✅ Checks Docker installation
- ✅ Opens interactive presentation for participants
- ✅ Runs guided setup with teaching pauses
- ✅ Launches your chosen interface (CLI/Web/Both)

### Option 2: Manual Configuration
```bash
# 1. Interactive setup (runs in container - no Python needed!)
docker-compose --profile setup run --rm setup

# 2. Start the workshop
docker-compose --profile local up --build              # CLI interface
docker-compose --profile local --profile web up --build # Web interface
```

### Option 3: Pre-configure Environment
```bash
# Copy and edit the example configuration
cp .env.example .env
# Edit .env with your settings, then:
./run_workshop.sh
```

The setup will guide you through:
- **Redis Configuration**: Choose local container or cloud instance
- **OpenAI Setup**: Enter your API key
- **Document Source**: Select Wikipedia URL, text file, or PDF

### 3. Ask Questions!

The application will process your document and allow you to ask questions about it.

## 📁 Project Structure

```
rag-workshop/
├── main.py              # Core RAG application
├── web_interface.py     # Streamlit web interface
├── setup.py            # Interactive setup script
├── docker-compose.yml  # Container orchestration
├── Dockerfile          # Application container
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── .env               # Configuration (created by setup)
└── data/              # Directory for local files
```

## 🔧 Configuration Options

### Redis Options

1. **Local Redis Container** (Recommended for workshop)
   - Automatically starts Redis with RedisSearch module
   - No external dependencies
   - Perfect for learning and development

2. **Cloud Redis Instance**
   - Use your existing Redis Cloud instance
   - Requires RedisSearch module enabled
   - Format: `redis://default:password@host:port`

### Document Sources

1. **Wikipedia URL** (Recommended)
   - Easy to use and demonstrate
   - Rich content for testing
   - Example: `https://en.wikipedia.org/wiki/Artificial_intelligence`

2. **Text File**
   - Upload any `.txt` file to the `data/` directory
   - Good for custom content

3. **PDF File**
   - Upload any `.pdf` file to the `data/` directory
   - Automatically extracts text content

## 🧠 How RAG Works - Step by Step

### Step 1: Document Processing
```python
# Extract text from source
text = extract_text_from_source(document_url)

# Split into manageable chunks
chunks = split_text_into_chunks(text, chunk_size=500)
```

### Step 2: Generate Embeddings
```python
# Convert text chunks to vectors
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)
```

### Step 3: Store in Vector Database
```python
# Store embeddings in Redis with metadata
redis_client.hset(chunk_id, {
    'text': chunk_text,
    'embedding': embedding_bytes,
    'source': document_source
})
```

### Step 4: Query Processing
```python
# Convert user question to embedding
query_embedding = model.encode(user_question)

# Find similar chunks
similar_chunks = redis_search(query_embedding, k=5)
```

### Step 5: Generate Response
```python
# Combine retrieved chunks as context
context = '\n'.join([chunk['text'] for chunk in similar_chunks])

# Generate answer using OpenAI
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Answer based on context"},
        {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
    ]
)
```

## 🎓 Workshop Exercises

### Exercise 1: Basic RAG
1. Set up the environment with a Wikipedia article
2. Ask simple factual questions
3. Observe how the system retrieves relevant chunks

### Exercise 2: Understanding Chunking
1. Experiment with different chunk sizes
2. See how chunk size affects retrieval quality
3. Understand the trade-offs

### Exercise 3: Embedding Exploration
1. Try different embedding models
2. Compare retrieval results
3. Understand semantic vs. keyword search

### Exercise 4: Advanced Queries
1. Ask complex, multi-part questions
2. Test edge cases and limitations
3. Explore prompt engineering techniques

## 🔍 Monitoring and Debugging

### Redis Insight (Local Redis)
- Access Redis Insight at: http://localhost:8001
- Visualize your vector index
- Monitor search performance

### Application Logs
- Watch Docker logs for detailed processing information
- Understand each step of the RAG pipeline

### Web Interface Features
- Real-time system status
- Conversation history
- Step-by-step RAG explanation

## 🚨 Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   # Check if Redis is running
   docker-compose ps
   
   # Restart Redis
   docker-compose restart redis
   ```

2. **OpenAI API Errors**
   - Verify your API key is correct
   - Check your OpenAI account has credits
   - Ensure API key has proper permissions

3. **Document Processing Errors**
   - Verify URLs are accessible
   - Check file paths are correct
   - Ensure files are readable

4. **Memory Issues**
   - Reduce chunk size for large documents
   - Use smaller embedding models
   - Process documents in batches

### Getting Help

1. Check the application logs
2. Verify your `.env` configuration
3. Test individual components
4. Ask the workshop instructor

## 📚 Additional Resources

- [Redis Vector Similarity](https://redis.io/docs/stack/search/reference/vectors/)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [RAG Best Practices](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)

## 🤝 Contributing

This workshop is designed to be educational and extensible. Feel free to:
- Add new document processors
- Experiment with different embedding models
- Implement advanced retrieval strategies
- Create additional interfaces

## 📄 License

This workshop material is provided for educational purposes. Please respect the terms of service for all external APIs and services used.

---

**Happy Learning! 🎉**

For questions or issues during the workshop, please ask your instructor or check the troubleshooting section above.
