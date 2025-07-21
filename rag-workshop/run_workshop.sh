#!/bin/bash

# RAG Workshop Runner Script
# This script helps participants easily start the workshop
# NO PYTHON INSTALLATION REQUIRED - Everything runs in containers!

set -e

echo "🚀 RAG Workshop Runner"
echo "====================="
echo "No Python installation required - everything runs in Docker!"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "✅ Docker is running"

# Check if setup has been run
if [ ! -f ".env" ]; then
    echo ""
    echo "⚙️  First time setup required..."
    echo "Running interactive configuration in container..."
    echo ""

    # Run setup in container
    docker-compose --profile setup run --rm setup

    if [ ! -f ".env" ]; then
        echo "❌ Setup was cancelled or failed. Exiting."
        exit 1
    fi

    echo "✅ Setup completed!"
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Determine which profile to use
PROFILE=${DOCKER_PROFILE:-local}

echo ""
echo "📋 Configuration Summary:"
echo "  Redis: ${REDIS_URL:-Not set}"
echo "  Document: ${DOCUMENT_SOURCE:-Not set}"
echo "  Profile: $PROFILE"
echo ""

# Ask user which interface they want
echo "Choose your workshop interface:"
echo "1. 🖥️  Command Line Interface (CLI) - Interactive terminal"
echo "2. 🌐 Web Interface (Streamlit) - Browser-based GUI"
echo "3. 🚀 Both interfaces - CLI + Web"

read -p "Enter choice (1-3): " interface_choice

echo ""
echo "🐳 Building and starting containers..."

case $interface_choice in
    1)
        echo "🖥️  Starting CLI interface..."
        echo "📝 The application will start in interactive mode"
        docker-compose --profile $PROFILE up --build
        ;;
    2)
        echo "🌐 Starting web interface..."
        echo "📝 Visit http://localhost:8501 in your browser"
        docker-compose --profile $PROFILE --profile web up --build
        ;;
    3)
        echo "🚀 Starting both interfaces..."
        echo "📝 CLI will run in terminal, web at http://localhost:8501"
        docker-compose --profile $PROFILE --profile web up --build
        ;;
    *)
        echo "❌ Invalid choice. Starting CLI interface..."
        docker-compose --profile $PROFILE up --build
        ;;
esac

echo ""
echo "🎉 Workshop session completed!"
echo "Thank you for participating in the RAG Workshop!"
