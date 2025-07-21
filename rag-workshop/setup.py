#!/usr/bin/env python3
"""
RAG Workshop Setup Script
Interactive configuration for the RAG workshop environment.
"""

import os
import sys
import time
import getpass
import subprocess
from pathlib import Path
from dotenv import load_dotenv, set_key
import requests

def print_header():
    """Print workshop header"""
    print("🚀 Welcome to the RAG Workshop Setup!")
    print("=" * 60)
    print("This workshop will teach you about Retrieval-Augmented Generation (RAG)")
    print("using Redis as a vector database and OpenAI for text generation.")
    print("=" * 60)

def validate_url(url):
    """Validate if URL is accessible"""
    try:
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except:
        return False

def validate_file_path(file_path):
    """Validate if file exists and is readable"""
    path = Path(file_path)
    return path.exists() and path.is_file()

def setup_redis():
    """Configure Redis connection"""
    print("\n📊 STEP 1: Redis Configuration")
    print("-" * 40)
    print("Redis will store our document embeddings as vectors.")
    print("You can use either:")
    print("  • Local Redis (we'll run it in Docker)")
    print("  • Cloud Redis (your existing instance)")

    while True:
        choice = input("\nUse local Redis container? (y/n): ").lower().strip()

        if choice == 'y':
            redis_url = "redis://localhost:6379"
            docker_profile = "local"
            print("✅ Using local Redis container")
            print("   We'll start Redis with RedisSearch module for vector similarity")
            break
        elif choice == 'n':
            print("\n🌐 Cloud Redis Configuration")
            print("Enter your complete Redis connection string.")
            print("Example: redis://default:password@host:port")

            while True:
                redis_url = input("Redis connection string: ").strip()
                if redis_url.startswith('redis://'):
                    docker_profile = "cloud"
                    print("✅ Using cloud Redis instance")
                    break
                else:
                    print("❌ Please enter a valid Redis URL starting with 'redis://'")
        else:
            print("❌ Please enter 'y' or 'n'")

    return redis_url, docker_profile

def setup_openai():
    """Configure OpenAI API"""
    print("\n🤖 STEP 2: OpenAI Configuration")
    print("-" * 40)
    print("We'll use OpenAI's GPT model to generate answers based on retrieved context.")

    while True:
        openai_key = getpass.getpass("Enter your OpenAI API key: ").strip()
        if openai_key:
            print("✅ OpenAI API key configured")
            break
        else:
            print("❌ API key cannot be empty")

    return openai_key

def setup_document():
    """Configure document source"""
    print("\n📄 STEP 3: Document Source")
    print("-" * 40)
    print("Choose what document you want to use for the RAG demonstration:")
    print("1. Wikipedia URL (recommended for demo)")
    print("2. Local text file")
    print("3. Local PDF file")

    while True:
        doc_choice = input("\nEnter choice (1-3): ").strip()

        if doc_choice == "1":
            print("\n🌐 Wikipedia URL Configuration")
            print("Example: https://en.wikipedia.org/wiki/Artificial_intelligence")

            while True:
                doc_source = input("Enter Wikipedia URL: ").strip()
                if doc_source.startswith('http'):
                    print("🔍 Validating URL...")
                    if validate_url(doc_source):
                        print("✅ URL is accessible")
                        break
                    else:
                        print("❌ URL is not accessible. Please check and try again.")
                else:
                    print("❌ Please enter a valid HTTP/HTTPS URL")
            break

        elif doc_choice == "2":
            print("\n📝 Text File Configuration")

            while True:
                doc_source = input("Enter path to text file: ").strip()
                if validate_file_path(doc_source):
                    print("✅ Text file found")
                    break
                else:
                    print("❌ File not found. Please check the path and try again.")
            break

        elif doc_choice == "3":
            print("\n📋 PDF File Configuration")

            while True:
                doc_source = input("Enter path to PDF file: ").strip()
                if validate_file_path(doc_source):
                    if doc_source.lower().endswith('.pdf'):
                        print("✅ PDF file found")
                        break
                    else:
                        print("❌ File doesn't appear to be a PDF. Please check the extension.")
                else:
                    print("❌ File not found. Please check the path and try again.")
            break

        else:
            print("❌ Please enter 1, 2, or 3")

    return doc_source, doc_choice

def create_data_directory():
    """Create data directory for local files"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print(f"✅ Created data directory: {data_dir.absolute()}")

def save_configuration(redis_url, openai_key, doc_source, doc_choice, docker_profile):
    """Save configuration to .env file"""
    print("\n💾 STEP 4: Saving Configuration")
    print("-" * 40)

    env_vars = {
        'REDIS_URL': redis_url,
        'OPENAI_API_KEY': openai_key,
        'DOCUMENT_SOURCE': doc_source,
        'DOCUMENT_TYPE': doc_choice,
        'DOCKER_PROFILE': docker_profile
    }

    # Create .env file
    env_file = Path('.env')
    for key, value in env_vars.items():
        set_key(str(env_file), key, value)

    print(f"✅ Configuration saved to {env_file.absolute()}")
    return env_vars

def print_next_steps(docker_profile):
    """Print instructions for next steps"""
    print("\n🎉 Setup Complete!")
    print("=" * 60)
    print("Your RAG workshop environment is now configured.")
    print("\nNext steps:")
    print("1. Start the application:")

    if docker_profile == "local":
        print("   docker-compose --profile local up --build")
        print("   (This will start both Redis and the RAG application)")
    else:
        print("   docker-compose --profile cloud up --build")
        print("   (This will start the RAG application with your cloud Redis)")

    print("\n2. Choose your interface:")
    print("   • CLI: The application will start automatically")
    print("   • Web: Add '--profile web' to the docker-compose command")
    print("     Then visit: http://localhost:8501")

    print("\n3. Workshop Learning Points:")
    print("   • Document processing and chunking")
    print("   • Embedding generation with sentence transformers")
    print("   • Vector storage and similarity search in Redis")
    print("   • Retrieval-augmented generation with OpenAI")

    print("\n📚 Files created:")
    print("   • .env - Your configuration")
    print("   • main.py - Core RAG application")
    print("   • web_interface.py - Streamlit web interface")
    print("   • docker-compose.yml - Container orchestration")

def pause_for_explanation(title, pause_time=10):
    """Pause for instructor explanation"""
    print(f"\n⏸️  INSTRUCTOR PAUSE: {title}")
    print("=" * 60)
    print("🎓 This is a teaching moment - instructor will explain concepts")
    print(f"⏱️  Pausing for {pause_time} seconds...")
    print("=" * 60)

    for i in range(pause_time, 0, -1):
        print(f"⏱️  {i} seconds remaining...", end="\r")
        time.sleep(1)

    print("\n✅ Continuing setup...")
    print()

def setup_environment():
    """Main setup function with teaching pauses"""
    print_header()

    # Teaching moment 1: RAG Overview
    pause_for_explanation("What is RAG and why are we building this?", 15)

    # Create data directory
    create_data_directory()

    # Teaching moment 2: Vector Databases
    pause_for_explanation("Understanding Vector Databases and Redis", 12)

    # Setup components
    redis_url, docker_profile = setup_redis()

    # Teaching moment 3: Embeddings and LLMs
    pause_for_explanation("How Embeddings and LLMs work together", 12)

    openai_key = setup_openai()

    # Teaching moment 4: Document Processing
    pause_for_explanation("Document Processing and Chunking Strategies", 10)

    doc_source, doc_choice = setup_document()

    # Save configuration
    config = save_configuration(redis_url, openai_key, doc_source, doc_choice, docker_profile)

    # Teaching moment 5: What happens next
    pause_for_explanation("The RAG Pipeline - What happens when we start", 8)

    # Print next steps
    print_next_steps(docker_profile)

    return config

if __name__ == "__main__":
    try:
        setup_environment()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)