#!/bin/bash

# RAG Workshop - Super Simple Starter
# Just run this script - no Python installation needed!

echo "🚀 Welcome to the RAG Workshop!"
echo "================================"
echo ""
echo "This workshop teaches Retrieval-Augmented Generation (RAG)"
echo "using Redis as a vector database and OpenAI for generation."
echo ""
echo "✅ No Python installation required"
echo "✅ Everything runs in Docker containers"
echo "✅ Interactive setup included"
echo ""

# Function to check if Docker is available and add to PATH if needed
check_docker() {
    # First, try to find docker in common locations
    DOCKER_PATHS=(
        "/usr/local/bin/docker"
        "/Applications/Docker.app/Contents/Resources/bin/docker"
        "/opt/homebrew/bin/docker"
        "$(which docker 2>/dev/null)"
    )

    for docker_path in "${DOCKER_PATHS[@]}"; do
        if [ -x "$docker_path" ]; then
            export PATH="$(dirname "$docker_path"):$PATH"
            echo "✅ Found Docker at: $docker_path"
            return 0
        fi
    done

    return 1
}

# Check Docker availability
echo "🔍 Checking Docker installation..."

if ! command -v docker >/dev/null 2>&1; then
    echo "⚠️  Docker command not found in PATH"
    echo "🔍 Searching for Docker in common locations..."

    if check_docker; then
        echo "✅ Docker found and added to PATH"
    else
        echo "❌ Docker not found!"
        echo ""
        echo "📥 Please install Docker Desktop:"
        echo "   1. Visit: https://www.docker.com/products/docker-desktop"
        echo "   2. Download Docker Desktop for Mac"
        echo "   3. Install and start Docker Desktop"
        echo "   4. Wait for Docker Desktop to fully start (whale icon in menu bar)"
        echo "   5. Run this script again"
        echo ""
        exit 1
    fi
fi

# Test Docker daemon
echo "🐳 Testing Docker connection..."
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker daemon is not running or accessible."
    echo ""
    echo "🔧 Troubleshooting steps:"
    echo "   1. Make sure Docker Desktop is running (whale icon in menu bar)"
    echo "   2. If Docker Desktop is running, try restarting it"
    echo "   3. Check if you need to accept Docker Desktop's terms"
    echo "   4. Try running: docker info"
    echo ""
    echo "💡 If Docker Desktop is installed but not working:"
    echo "   - Open Docker Desktop application manually"
    echo "   - Wait for it to fully start (may take 1-2 minutes)"
    echo "   - Look for the whale icon in your menu bar"
    echo "   - Then run this script again"
    echo ""
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Quick setup check
if [ ! -f ".env" ]; then
    echo "🔧 First-time setup needed..."
    echo "   We'll configure Redis, OpenAI, and document source"
    echo ""
    read -p "Press Enter to start interactive setup..."
    
    # Run containerized setup
    docker-compose --profile setup run --rm setup
    
    if [ ! -f ".env" ]; then
        echo "❌ Setup cancelled. Exiting."
        exit 1
    fi
fi

echo "🎯 Starting RAG Workshop..."
echo ""
echo "Choose your interface:"
echo "  1) 🖥️  Terminal (CLI) - Interactive command line"
echo "  2) 🌐 Web Browser - Visual interface at http://localhost:8501"
echo ""

read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo "🖥️  Starting terminal interface..."
        docker-compose --profile local up --build
        ;;
    2)
        echo "🌐 Starting web interface..."
        echo "📝 Open http://localhost:8501 in your browser"
        docker-compose --profile local --profile web up --build
        ;;
    *)
        echo "🖥️  Starting terminal interface (default)..."
        docker-compose --profile local up --build
        ;;
esac

echo ""
echo "🎉 Workshop completed! Thanks for participating!"
