#!/bin/bash

# RAG Workshop - One Command to Rule Them All
# This script handles everything: Docker check, setup, and launch
# Usage: ./workshop.sh [--fresh]  (--fresh removes existing configuration)

# Check for --fresh flag
FRESH_START=false
if [ "$1" = "--fresh" ]; then
    FRESH_START=true
    echo "Fresh start requested - removing existing configuration"
    rm -f .env
fi

echo "RAG Workshop - Complete Setup & Launch"
echo "=========================================="
echo ""
echo "This workshop teaches Retrieval-Augmented Generation (RAG)"
echo "- No Python installation required"
echo "- Everything runs in Docker containers"
echo "- Includes instructor teaching pauses"
echo ""

# Function to check Docker (same as before)
check_docker() {
    DOCKER_PATHS=(
        "/usr/local/bin/docker"
        "/Applications/Docker.app/Contents/Resources/bin/docker"
        "/opt/homebrew/bin/docker"
        "$(which docker 2>/dev/null)"
    )
    
    for docker_path in "${DOCKER_PATHS[@]}"; do
        if [ -x "$docker_path" ]; then
            export PATH="$(dirname "$docker_path"):$PATH"
            echo "Found Docker at: $docker_path"
            return 0
        fi
    done
    
    return 1
}

# Check Docker
echo "Checking Docker installation..."
if ! command -v docker >/dev/null 2>&1; then
    echo "Docker command not found in PATH"
    if check_docker; then
        echo "Docker found and added to PATH"
    else
        echo "Docker not found!"
        echo ""
        echo "Please install Docker Desktop first:"
        echo "   Mac: https://www.docker.com/products/docker-desktop"
        echo "   Then run this script again"
        exit 1
    fi
fi

# Test Docker daemon
echo "Testing Docker connection..."
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running"
    echo ""
    echo "Please start Docker Desktop:"
    echo "   1. Open Docker Desktop application"
    echo "   2. Wait for whale icon in menu bar"
    echo "   3. Run this script again"
    exit 1
fi

echo "Docker is ready!"
echo ""

# Open workshop presentation for participants
echo "Opening workshop presentation for participants..."
if command -v open >/dev/null 2>&1; then
    # macOS
    open workshop_presentation.html
elif command -v xdg-open >/dev/null 2>&1; then
    # Linux
    xdg-open workshop_presentation.html
elif command -v start >/dev/null 2>&1; then
    # Windows
    start workshop_presentation.html
else
    echo "Please open workshop_presentation.html in your browser"
fi
echo ""

# Check if already configured
if [ -f ".env" ]; then
    echo "Found existing configuration"
    source .env
    echo "   Redis: ${REDIS_URL}"
    echo "   Document: ${DOCUMENT_SOURCE}"
    echo ""
    
    read -p "Use existing configuration? (y/n): " use_existing
    if [ "$use_existing" != "y" ]; then
        echo "Running new setup..."
        rm .env
    fi
fi

# Run setup if needed
if [ ! -f ".env" ]; then
    echo "Starting Interactive Workshop Setup"
    echo "   (Includes instructor teaching pauses)"
    echo ""

    # Run containerized setup with teaching pauses
    docker-compose --profile setup run --rm setup

    if [ ! -f ".env" ]; then
        echo "Setup was cancelled. Exiting."
        exit 1
    fi
fi

# Load configuration
source .env
PROFILE=${DOCKER_PROFILE:-local}

echo ""
echo "Launching RAG Workshop!"
echo "=========================="
echo ""
echo "Choose your interface:"
echo "  1) Terminal (CLI) - See detailed logs and process"
echo "  2) Web Browser - Visual interface with explanations"
echo "  3) Both - Terminal + Web (recommended for instructors)"
echo ""

read -p "Enter choice (1-3): " interface_choice

echo ""
echo "Preparing workshop environment..."
echo "   - Stopping any existing containers"
echo "   - Building fresh containers with latest code"
echo "   - This ensures a clean workshop experience"
echo ""

# Stop any existing containers and remove old images
echo "Stopping existing containers..."
docker-compose down --remove-orphans 2>/dev/null || true

# Remove old workshop images to force complete rebuild
echo "Removing old workshop images..."
docker images | grep rag-workshop | awk '{print $3}' | xargs docker rmi -f 2>/dev/null || true

# Build containers with no cache to ensure latest code
echo "Building containers (this may take a few minutes)..."
docker-compose build --no-cache

echo ""
echo "Starting workshop containers..."
echo ""

case $interface_choice in
    1)
        echo "Starting CLI interface..."
        echo "You'll be able to ask questions in the terminal"
        docker-compose --profile $PROFILE up
        ;;
    2)
        echo "Starting web interface..."
        echo "Open http://localhost:8501 in your browser"
        docker-compose --profile $PROFILE --profile web up
        ;;
    3)
        echo "Starting both interfaces..."
        echo "Terminal: Interactive CLI"
        echo "Browser: http://localhost:8501"
        echo "Redis Insight: http://localhost:8001 (if using local Redis)"
        docker-compose --profile $PROFILE --profile web up
        ;;
    *)
        echo "Starting CLI interface (default)..."
        docker-compose --profile $PROFILE up
        ;;
esac

echo ""
echo "Workshop session completed!"
echo ""
echo "What you learned:"
echo "   - Document processing and chunking"
echo "   - Embedding generation for semantic search"
echo "   - Vector storage with Redis"
echo "   - Retrieval-augmented generation with OpenAI"
echo ""
echo "Next steps:"
echo "   - Run again: ./workshop.sh"
echo "   - Clean shutdown: ./stop.sh"
echo "   - Quick stop: docker-compose down"
echo ""
echo "Thank you for participating in the RAG Workshop!"
