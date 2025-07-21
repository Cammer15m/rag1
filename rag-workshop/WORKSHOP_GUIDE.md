# 🎓 RAG Workshop - Instructor Guide

This guide provides detailed explanations for each step of the RAG workshop, designed to help instructors explain the concepts and demonstrate the system to participants.

## 📋 Workshop Agenda (90 minutes)

### Part 1: Introduction (15 minutes)
- What is RAG and why is it important?
- Overview of the system architecture
- Setup and configuration walkthrough

### Part 2: Document Processing (20 minutes)
- Text extraction and cleaning
- Chunking strategies and trade-offs
- Hands-on: Process a Wikipedia article

### Part 3: Embeddings and Vector Storage (25 minutes)
- Understanding embeddings and semantic similarity
- Redis as a vector database
- Hands-on: Generate and store embeddings

### Part 4: Retrieval and Generation (20 minutes)
- Similarity search mechanics
- Context preparation for LLMs
- Hands-on: Ask questions and see retrieval

### Part 5: Advanced Topics and Q&A (10 minutes)
- Optimization strategies
- Production considerations
- Open discussion

## 🔍 Detailed Step-by-Step Explanations

### Step 1: Document Processing Deep Dive

#### What happens when we process a document?

1. **Text Extraction**
   ```python
   # For Wikipedia URLs
   response = requests.get(url)
   soup = BeautifulSoup(response.content, 'html.parser')
   text = soup.get_text()
   ```
   
   **Explain to participants:**
   - We're extracting raw text from HTML
   - Removing navigation, ads, and other non-content elements
   - This gives us clean, readable text to work with

2. **Text Chunking**
   ```python
   def _chunk_text(self, text: str, chunk_size: int = 500):
       chunks = []
       start = 0
       while start < len(text):
           end = start + chunk_size
           # Try to break at sentence boundary
           chunk_text = text[start:end]
           chunks.append(chunk_text)
           start = end - overlap
   ```
   
   **Key concepts to explain:**
   - **Why chunk?** LLMs have context limits, embeddings work better on focused text
   - **Chunk size trade-offs:** 
     - Small chunks (200-300 words): More precise, but may lose context
     - Large chunks (800-1000 words): More context, but less precise retrieval
   - **Overlap:** Ensures important information isn't split across chunks

#### Demonstration Points:
- Show the original Wikipedia page
- Display the extracted text
- Show how it's split into chunks
- Explain the chunk metadata (ID, source, index)

### Step 2: Embeddings Generation Deep Dive

#### What are embeddings?

**Simple explanation:**
"Embeddings convert text into numbers that capture meaning. Similar text gets similar numbers."

#### The Process:
```python
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(text_chunks)
```

**Key concepts:**
- **Vector space:** Each chunk becomes a point in 384-dimensional space
- **Semantic similarity:** Related concepts cluster together
- **Model choice:** Different models capture different aspects of meaning

#### Demonstration:
1. Show a few example chunks
2. Explain that each becomes a 384-number vector
3. Demonstrate that similar chunks have similar vectors
4. Use the web interface to show embedding generation progress

### Step 3: Vector Storage in Redis

#### Why Redis for vectors?

**Explain the benefits:**
- **Speed:** In-memory storage for fast retrieval
- **Scalability:** Can handle millions of vectors
- **Search capabilities:** Built-in similarity search with RedisSearch
- **Flexibility:** Can store metadata alongside vectors

#### The Storage Process:
```python
# Store each chunk with its embedding
redis_client.hset(chunk_id, {
    'text': chunk_text,
    'embedding': embedding_bytes,
    'source': document_source,
    'chunk_index': index
})
```

#### Index Creation:
```python
# Create a search index for fast similarity search
schema = [
    TextField("text"),
    VectorField("embedding", "HNSW", {
        "TYPE": "FLOAT32",
        "DIM": 384,
        "DISTANCE_METRIC": "COSINE"
    })
]
```

**Key concepts to explain:**
- **HNSW algorithm:** Hierarchical Navigable Small World graphs for fast approximate search
- **Cosine similarity:** Measures angle between vectors, good for text similarity
- **Index vs. storage:** Index enables fast search, storage holds the actual data

#### Demonstration:
- Show Redis Insight interface (http://localhost:8001)
- Display the stored chunks and their metadata
- Explain the index structure

### Step 4: Query Processing and Retrieval

#### The Query Journey:

1. **User asks a question:** "What is machine learning?"

2. **Question → Embedding:**
   ```python
   query_embedding = model.encode("What is machine learning?")
   ```

3. **Similarity Search:**
   ```python
   # Find the 5 most similar chunks
   results = redis_search(query_embedding, k=5)
   ```

4. **Context Preparation:**
   ```python
   context = '\n'.join([chunk['text'] for chunk in results])
   ```

5. **LLM Generation:**
   ```python
   prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
   response = openai.chat.completions.create(...)
   ```

#### Key concepts:
- **Semantic search vs. keyword search:** Find meaning, not just words
- **Relevance scoring:** Cosine similarity scores (0-1, higher = more similar)
- **Context window:** How much retrieved text we can fit in the LLM prompt
- **Prompt engineering:** How we structure the prompt affects the answer quality

#### Demonstration:
1. Ask a question in the interface
2. Show the retrieved chunks and their similarity scores
3. Display the final prompt sent to OpenAI
4. Show the generated response

### Step 5: Advanced Concepts

#### Optimization Strategies:

1. **Chunk Size Optimization**
   - Test different sizes for your use case
   - Consider document structure (paragraphs, sections)

2. **Embedding Model Selection**
   - General purpose: `all-MiniLM-L6-v2`
   - Domain-specific: Fine-tuned models
   - Multilingual: `paraphrase-multilingual-MiniLM-L12-v2`

3. **Retrieval Strategies**
   - **Hybrid search:** Combine semantic + keyword search
   - **Re-ranking:** Use a second model to re-rank results
   - **Query expansion:** Rephrase queries for better retrieval

4. **Production Considerations**
   - **Caching:** Cache embeddings and frequent queries
   - **Monitoring:** Track retrieval quality and response times
   - **Updates:** Handle document updates and deletions
   - **Security:** Protect API keys and user data

## 🎯 Interactive Exercises

### Exercise 1: Chunk Size Experiment
1. Modify `chunk_size` in the code
2. Reprocess the same document
3. Ask the same questions
4. Compare answer quality

### Exercise 2: Different Documents
1. Try a technical Wikipedia article
2. Try a historical article
3. Compare how well the system handles different content types

### Exercise 3: Query Variations
1. Ask the same question in different ways
2. Try very specific vs. general questions
3. Test edge cases (questions not in the document)

### Exercise 4: Embedding Exploration
1. Use Redis Insight to explore stored vectors
2. Look at similarity scores for different queries
3. Understand what makes chunks "similar"

## 🗣️ Key Talking Points

### Why RAG is Important:
- **Accuracy:** Grounds LLM responses in factual data
- **Freshness:** Can use up-to-date information
- **Transparency:** Can show sources for answers
- **Cost:** More efficient than fine-tuning for specific domains

### Real-World Applications:
- **Customer support:** Answer questions from documentation
- **Research:** Query large document collections
- **Education:** Interactive learning from textbooks
- **Legal:** Search through case law and regulations

### Limitations and Challenges:
- **Retrieval quality:** Bad retrieval = bad answers
- **Context limits:** Can't retrieve everything relevant
- **Hallucination:** LLM might still make things up
- **Complexity:** Many components that can fail

## 🔧 Troubleshooting Common Issues

### "No relevant chunks found"
- Check if document was processed correctly
- Verify embeddings were generated
- Try rephrasing the question

### "Redis connection failed"
- Ensure Docker containers are running
- Check Redis URL configuration
- Verify network connectivity

### "OpenAI API error"
- Check API key validity
- Verify account has credits
- Check rate limits

### Poor answer quality
- Experiment with chunk size
- Try different retrieval parameters (k value)
- Improve prompt engineering

## 📊 Success Metrics

By the end of the workshop, participants should be able to:
- [ ] Explain what RAG is and why it's useful
- [ ] Understand the role of each component (chunking, embeddings, vector DB, LLM)
- [ ] Set up and run a basic RAG system
- [ ] Identify optimization opportunities
- [ ] Troubleshoot common issues

## 🎉 Wrap-up Discussion

### Questions to ask participants:
1. What surprised you most about how RAG works?
2. What would you use this for in your work?
3. What challenges do you foresee in production?
4. What would you want to improve about this system?

### Next Steps:
- Experiment with your own documents
- Try different embedding models
- Explore advanced retrieval strategies
- Consider production deployment options
