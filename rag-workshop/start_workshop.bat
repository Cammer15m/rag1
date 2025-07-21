@echo off
REM RAG Workshop - Windows Starter Script
REM Just double-click this file - no Python installation needed!

echo 🚀 Welcome to the RAG Workshop!
echo ================================
echo.
echo This workshop teaches Retrieval-Augmented Generation (RAG)
echo using Redis as a vector database and OpenAI for generation.
echo.
echo ✅ No Python installation required
echo ✅ Everything runs in Docker containers
echo ✅ Interactive setup included
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running.
    echo    Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo ✅ Docker is running
echo.

REM Check for configuration
if not exist ".env" (
    echo 🔧 First-time setup needed...
    echo    We'll configure Redis, OpenAI, and document source
    echo.
    pause
    
    REM Run containerized setup
    docker-compose --profile setup run --rm setup
    
    if not exist ".env" (
        echo ❌ Setup cancelled. Exiting.
        pause
        exit /b 1
    )
)

echo 🎯 Starting RAG Workshop...
echo.
echo Choose your interface:
echo   1^) 🖥️  Terminal ^(CLI^) - Interactive command line
echo   2^) 🌐 Web Browser - Visual interface at http://localhost:8501
echo.

set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    echo 🖥️  Starting terminal interface...
    docker-compose --profile local up --build
) else if "%choice%"=="2" (
    echo 🌐 Starting web interface...
    echo 📝 Open http://localhost:8501 in your browser
    docker-compose --profile local --profile web up --build
) else (
    echo 🖥️  Starting terminal interface ^(default^)...
    docker-compose --profile local up --build
)

echo.
echo 🎉 Workshop completed! Thanks for participating!
pause
