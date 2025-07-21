# 👨‍🏫 Instructor Pre-Workshop Checklist

## 📋 Before the Workshop

### 1. Test the Complete Setup (30 minutes before)
```bash
# Test the full workshop flow
./start_workshop.sh

# Verify all components work:
# - Interactive setup
# - Document processing
# - Query interface
# - Web interface (if using)
```

### 2. Prepare Demo Environment
- [ ] Have a good Wikipedia article ready (e.g., Artificial Intelligence)
- [ ] Test with sample questions
- [ ] Ensure stable internet connection
- [ ] Have backup documents ready

### 3. Check System Requirements
- [ ] Docker Desktop installed and running
- [ ] Valid OpenAI API key with credits
- [ ] Sufficient disk space (2-3 GB for containers)
- [ ] Stable internet for downloading containers

## 🍎 Mac-Specific Preparation

### Common Mac Issues to Address

1. **Docker PATH Issues**
   - Our script auto-detects Docker in common locations
   - Have backup manual PATH commands ready
   - Know where Docker Desktop installs: `/Applications/Docker.app/`

2. **Permission Issues**
   - Docker Desktop may need admin password
   - Participants should run Docker Desktop manually first
   - Check that whale icon appears in menu bar

3. **First-Time Docker Users**
   - Docker Desktop setup wizard must be completed
   - Terms and conditions must be accepted
   - May need to restart after installation

### Pre-Workshop Instructions for Mac Participants

**Send this 24 hours before workshop:**

```
🍎 Mac Users - Please Complete Before Workshop:

1. Install Docker Desktop:
   - Visit: https://www.docker.com/products/docker-desktop
   - Download "Mac with Apple chip" or "Mac with Intel chip"
   - Install and start Docker Desktop
   - Complete the setup wizard

2. Verify installation:
   - Open Terminal
   - Run: docker --version
   - Should show Docker version info

3. If you have issues:
   - See MAC_SETUP.md in the workshop materials
   - Or arrive 15 minutes early for help

Need help? Contact [instructor email]
```

## 🎯 Workshop Flow Recommendations

### Opening (5 minutes)
1. **Quick Docker check**: Have everyone run `docker --version`
2. **Address issues immediately**: Help anyone with Docker problems
3. **Set expectations**: Explain the containerized approach

### Setup Phase (10 minutes)
1. **Demonstrate setup**: Show the interactive configuration
2. **Help with API keys**: Guide OpenAI key entry
3. **Choose demo document**: Use same document for everyone initially

### Teaching Phase (60 minutes)
1. **Use web interface for demos**: Visual and easier to follow
2. **Show terminal logs**: Explain what's happening behind the scenes
3. **Encourage experimentation**: Let participants try different questions

### Troubleshooting Phase (15 minutes)
1. **Common issues**: Address Docker, API, or connection problems
2. **Individual help**: Assist participants with specific issues
3. **Advanced exploration**: For those who finish early

## 🔧 Common Issues & Quick Fixes

### Docker Issues
```bash
# Check if Docker is running
docker info

# Restart Docker Desktop
# (Use GUI: Docker Desktop → Restart)

# Check container status
docker-compose ps

# Restart workshop containers
docker-compose down
./start_workshop.sh
```

### API Issues
```bash
# Check .env file
cat .env

# Re-run setup to fix configuration
docker-compose --profile setup run --rm setup
```

### Performance Issues
```bash
# Check Docker resource usage
# (Docker Desktop → Settings → Resources)

# Clean up old containers
docker system prune
```

## 📚 Teaching Points to Emphasize

### 1. Containerization Benefits
- **Consistency**: Same environment for everyone
- **Isolation**: No conflicts with existing software
- **Portability**: Works on any system with Docker
- **Production-ready**: Mirrors real deployment scenarios

### 2. RAG Concepts
- **Document chunking**: Why and how we split text
- **Embeddings**: Converting text to numbers for similarity
- **Vector search**: Finding relevant information
- **Context injection**: How we provide context to LLMs

### 3. Real-World Applications
- **Customer support**: Answering questions from documentation
- **Research**: Querying large document collections
- **Education**: Interactive learning from textbooks
- **Legal/Medical**: Searching specialized knowledge bases

## 🎯 Success Metrics

### Technical Success
- [ ] All participants can start the workshop
- [ ] Document processing completes successfully
- [ ] Queries return relevant answers
- [ ] No major technical blockers

### Learning Success
- [ ] Participants understand RAG concept
- [ ] Can explain the pipeline steps
- [ ] Successfully ask and get answers to questions
- [ ] Understand practical applications

## 🆘 Emergency Backup Plans

### If Docker Issues Persist
1. **Use instructor machine**: Demo on your working setup
2. **Pair programming**: Have participants work in pairs
3. **Cloud alternative**: Use Google Colab notebook (prepare backup)

### If Internet Issues
1. **Local documents**: Use pre-loaded text files
2. **Cached containers**: Pre-pull Docker images
3. **Offline mode**: Focus on concepts with slides

### If OpenAI Issues
1. **Shared API key**: Use instructor's key for demos
2. **Mock responses**: Show pre-recorded examples
3. **Focus on retrieval**: Demonstrate vector search without generation

## 📝 Post-Workshop

### Cleanup Instructions for Participants
```bash
# Stop all containers
docker-compose down

# Optional: Remove containers to free space
docker-compose down --rmi all --volumes

# Keep the code for future reference
# (Don't delete the workshop folder)
```

### Follow-up Resources
- Provide links to advanced RAG tutorials
- Share production deployment guides
- Offer office hours for questions
- Create a discussion forum or Slack channel

---

## 🎉 You're Ready to Teach!

This checklist ensures a smooth workshop experience. The containerized approach eliminates most technical issues, but being prepared for Mac-specific Docker challenges will help you assist participants quickly and keep the workshop on track.
