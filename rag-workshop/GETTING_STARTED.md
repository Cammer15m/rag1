# 🚀 RAG Workshop - Getting Started

## ✨ No Python Installation Required!

This workshop runs entirely in Docker containers. You only need:
- **Docker Desktop** installed and running
- **OpenAI API key** (we'll help you configure this)

### 🍎 Mac Users - Important!
**First time?** See [MAC_SETUP.md](MAC_SETUP.md) for detailed Docker installation instructions.

**Quick check**: Open Terminal and run `docker --version`
- ✅ **Works?** You're ready!
- ❌ **Error?** Follow the Mac setup guide first

## 🎯 Super Quick Start

### For Mac/Linux Users:
```bash
./start_workshop.sh
```

### For Windows Users:
Double-click: `start_workshop.bat`

That's it! The script will:
1. Check if Docker is running
2. Guide you through interactive setup (if needed)
3. Start the workshop interface

## 🔧 What the Setup Does

The interactive setup will ask you:

### 1. Redis Configuration
- **Local Redis** (recommended): We'll start a Redis container for you
- **Cloud Redis**: Use your existing Redis Cloud instance

### 2. OpenAI API Key
- Enter your OpenAI API key securely
- The key is stored locally in a `.env` file

### 3. Document Source
Choose what to analyze:
- **Wikipedia URL** (great for demos): e.g., https://en.wikipedia.org/wiki/Artificial_intelligence
- **Text file**: Upload a `.txt` file to the `data/` folder
- **PDF file**: Upload a `.pdf` file to the `data/` folder

## 🖥️ Interface Options

### Terminal Interface (CLI)
- Interactive command-line experience
- Perfect for understanding the step-by-step process
- Shows detailed logging and processing information

### Web Interface (Browser)
- User-friendly graphical interface
- Visit: http://localhost:8501
- Great for demonstrations and visual learners
- Shows system status and conversation history

## 📚 What You'll Learn

### 1. Document Processing (20 minutes)
- How to extract text from different sources
- Text chunking strategies and why they matter
- Preprocessing and cleaning techniques

### 2. Embeddings & Vector Storage (25 minutes)
- Converting text to numerical vectors
- Understanding semantic similarity
- Using Redis as a vector database

### 3. Retrieval & Generation (20 minutes)
- Finding relevant information for queries
- Combining retrieved context with LLMs
- Generating accurate, contextual responses

### 4. Hands-on Practice (25 minutes)
- Ask questions about your document
- Experiment with different query types
- Understand system limitations and strengths

## 🔍 Sample Questions to Try

Once your document is processed, try asking:

**For AI/Technology documents:**
- "What is artificial intelligence?"
- "What are the main applications of AI?"
- "What are the challenges in AI development?"

**For any document:**
- "Summarize the main points"
- "What are the key concepts discussed?"
- "Can you explain [specific topic] in simple terms?"

## 🛠️ Troubleshooting

### Docker Issues
```bash
# Check if Docker is running
docker info

# If not running, start Docker Desktop
```

### Setup Issues
```bash
# Re-run setup if needed
docker-compose --profile setup run --rm setup

# Or delete .env and start over
rm .env
./start_workshop.sh
```

### Connection Issues
```bash
# Check container status
docker-compose ps

# Restart everything
docker-compose down
./start_workshop.sh
```

## 🎓 Workshop Tips

### For Instructors:
- Use the web interface for demonstrations
- Show the Redis Insight UI at http://localhost:8001
- Explain each step as the system processes documents
- Use the terminal interface to show detailed logs

### For Participants:
- Don't worry about the technical details initially
- Focus on understanding the RAG concept
- Experiment with different types of questions
- Ask about anything you don't understand

## 📁 Files You'll See

After setup, you'll have:
- `.env` - Your configuration (keep this private!)
- `data/` - Folder for your documents
- Docker containers running the RAG system

## 🚀 Advanced Usage

### Manual Docker Commands
```bash
# Setup only
docker-compose --profile setup run --rm setup

# Local Redis + CLI
docker-compose --profile local up --build

# Local Redis + Web interface
docker-compose --profile local --profile web up --build

# Cloud Redis
docker-compose --profile cloud up --build
```

### Testing the System
```bash
# Run automated tests
docker-compose run --rm rag-app-local python test_rag.py
```

## 🎉 Ready to Start?

1. **Make sure Docker Desktop is running**
2. **Run the startup script for your platform**
3. **Follow the interactive setup**
4. **Start asking questions!**

The workshop is designed to be educational and fun. Don't hesitate to experiment and ask questions!

---

**Need help?** Ask your instructor or check the troubleshooting section above.
