# 🍎 Mac Setup Guide for RAG Workshop

## 📋 What You Need to Install

**Only one thing: Docker Desktop for Mac**

## 🚀 Step-by-Step Installation

### Step 1: Download Docker Desktop
1. Visit: https://www.docker.com/products/docker-desktop
2. Click "Download for Mac"
3. Choose the right version:
   - **Apple Silicon (M1/M2/M3)**: Download "Mac with Apple chip"
   - **Intel Mac**: Download "Mac with Intel chip"

### Step 2: Install Docker Desktop
1. Open the downloaded `.dmg` file
2. Drag Docker to your Applications folder
3. Open Docker Desktop from Applications
4. **Important**: Complete the initial setup and accept terms

### Step 3: Verify Installation
1. Wait for Docker Desktop to fully start
2. Look for the **whale icon** in your menu bar (top right)
3. The whale icon should be **steady** (not animated)

### Step 4: Test Docker
Open Terminal and run:
```bash
docker --version
```

You should see something like:
```
Docker version 24.0.6, build ed223bc
```

## 🔧 Common Mac Issues & Solutions

### Issue 1: "docker: command not found"
**Cause**: Docker not in PATH or not fully started

**Solutions**:
1. **Wait longer**: Docker Desktop can take 2-3 minutes to fully start
2. **Restart Docker Desktop**: Quit and reopen the application
3. **Check menu bar**: Make sure whale icon is present and steady
4. **Restart Terminal**: Close and reopen Terminal after Docker starts

### Issue 2: "Cannot connect to the Docker daemon"
**Cause**: Docker Desktop not running or permission issues

**Solutions**:
1. **Start Docker Desktop**: Open from Applications folder
2. **Check permissions**: Docker may ask for admin password
3. **Complete setup**: Make sure you've accepted all terms and completed setup
4. **Restart Mac**: Sometimes needed after first install

### Issue 3: Docker Desktop won't start
**Cause**: Various system issues

**Solutions**:
1. **Check system requirements**: macOS 10.15 or newer
2. **Free up disk space**: Docker needs at least 4GB free
3. **Update macOS**: Ensure you have latest updates
4. **Reinstall Docker**: Delete and reinstall if needed

### Issue 4: "Permission denied" errors
**Cause**: User not in docker group or admin rights needed

**Solutions**:
1. **Run Docker Desktop as admin**: Right-click → "Open"
2. **Enter admin password** when prompted
3. **Check Docker Desktop settings**: Ensure it's configured properly

## 🎯 Workshop-Specific Setup

### Our Script Handles Most Issues
The `start_workshop.sh` script automatically:
- Searches for Docker in common Mac locations
- Adds Docker to PATH if needed
- Provides helpful error messages
- Guides you through troubleshooting

### Manual PATH Fix (if needed)
If Docker is installed but not found, add to your shell profile:

**For zsh (default on newer Macs)**:
```bash
echo 'export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**For bash**:
```bash
echo 'export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"' >> ~/.bash_profile
source ~/.bash_profile
```

## ✅ Verification Checklist

Before starting the workshop, ensure:

- [ ] Docker Desktop is installed
- [ ] Docker Desktop is running (whale icon in menu bar)
- [ ] `docker --version` works in Terminal
- [ ] `docker info` shows Docker daemon info
- [ ] No permission errors when running Docker commands

## 🚀 Ready for Workshop

Once Docker is working, start the workshop:

```bash
# Navigate to workshop folder
cd rag-workshop

# Start the workshop
./start_workshop.sh
```

## 🆘 Still Having Issues?

### Quick Diagnostic
Run this in Terminal:
```bash
# Check if Docker is installed
ls -la /Applications/Docker.app

# Check if Docker is in PATH
which docker

# Check if Docker daemon is running
docker info
```

### Get Help
1. **During workshop**: Ask your instructor
2. **Docker issues**: Check Docker Desktop troubleshooting docs
3. **Mac-specific**: Check Apple's support for your macOS version

## 💡 Pro Tips for Mac Users

1. **Keep Docker Desktop running**: Don't quit it during the workshop
2. **Monitor resources**: Docker Desktop shows CPU/memory usage
3. **Use Activity Monitor**: Check if Docker processes are running
4. **Terminal alternatives**: iTerm2 or built-in Terminal both work fine
5. **Multiple terminals**: You can open multiple Terminal windows if needed

## 🔄 Alternative Installation Methods

### Homebrew (Advanced Users)
```bash
# Install Docker via Homebrew
brew install --cask docker

# Start Docker Desktop
open /Applications/Docker.app
```

### Command Line Tools Only (Not Recommended for Workshop)
```bash
# Install Docker CLI only (missing Docker Desktop features)
brew install docker
```

**Note**: For this workshop, we recommend Docker Desktop as it includes the GUI and is easier to troubleshoot.

---

## 🎉 You're Ready!

Once Docker Desktop is running and you can run `docker --version`, you're all set for the RAG workshop. The workshop script will handle everything else!
