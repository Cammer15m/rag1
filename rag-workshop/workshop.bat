@echo off
REM RAG Workshop - One Command to Rule Them All (Windows)
REM This script handles everything: Docker check, setup, and launch

echo 🚀 RAG Workshop - Complete Setup ^& Launch
echo ==========================================
echo.
echo This workshop teaches Retrieval-Augmented Generation ^(RAG^)
echo ✅ No Python installation required
echo ✅ Everything runs in Docker containers
echo ✅ Includes instructor teaching pauses
echo.

REM Check Docker
echo 🔍 Checking Docker installation...
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running
    echo.
    echo 🔧 Please start Docker Desktop:
    echo    1. Open Docker Desktop application
    echo    2. Wait for it to fully start
    echo    3. Run this script again
    echo.
    pause
    exit /b 1
)

echo ✅ Docker is ready!
echo.

REM Check existing configuration
if exist ".env" (
    echo 📋 Found existing configuration
    echo.
    set /p use_existing="Use existing configuration? (y/n): "
    if not "!use_existing!"=="y" (
        echo 🔄 Running new setup...
        del .env
    )
)

REM Run setup if needed
if not exist ".env" (
    echo 🎓 Starting Interactive Workshop Setup
    echo    ^(Includes instructor teaching pauses^)
    echo.
    
    REM Run containerized setup
    docker-compose --profile setup run --rm setup
    
    if not exist ".env" (
        echo ❌ Setup was cancelled. Exiting.
        pause
        exit /b 1
    )
)

echo.
echo 🚀 Launching RAG Workshop!
echo ==========================
echo.
echo Choose your interface:
echo   1^) 🖥️  Terminal ^(CLI^) - See detailed logs and process
echo   2^) 🌐 Web Browser - Visual interface with explanations
echo   3^) 🔬 Both - Terminal + Web ^(recommended for instructors^)
echo.

set /p interface_choice="Enter choice (1-3): "

echo.
echo 🐳 Building and starting containers...
echo    ^(This may take a few minutes on first run^)
echo.

if "%interface_choice%"=="1" (
    echo 🖥️  Starting CLI interface...
    echo 📝 You'll be able to ask questions in the terminal
    docker-compose --profile local up --build
) else if "%interface_choice%"=="2" (
    echo 🌐 Starting web interface...
    echo 📝 Open http://localhost:8501 in your browser
    docker-compose --profile local --profile web up --build
) else if "%interface_choice%"=="3" (
    echo 🔬 Starting both interfaces...
    echo 📝 Terminal: Interactive CLI
    echo 📝 Browser: http://localhost:8501
    echo 📝 Redis Insight: http://localhost:8001 ^(if using local Redis^)
    docker-compose --profile local --profile web up --build
) else (
    echo 🖥️  Starting CLI interface ^(default^)...
    docker-compose --profile local up --build
)

echo.
echo 🎉 Workshop session completed!
echo.
echo 📚 What you learned:
echo    ✅ Document processing and chunking
echo    ✅ Embedding generation for semantic search
echo    ✅ Vector storage with Redis
echo    ✅ Retrieval-augmented generation with OpenAI
echo.
echo 🔄 To run again: workshop.bat
echo 🧹 To clean up: docker-compose down
echo.
echo Thank you for participating in the RAG Workshop!
pause
