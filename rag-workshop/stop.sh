#!/bin/bash

# RAG Workshop - Stop Script
# This script stops and cleans up all workshop resources

echo "=================================================="
echo "RAG Workshop - Cleanup Script"
echo "=================================================="

# Function to print status
print_status() {
    echo "[$1] $2"
}

# Stop and remove all containers
print_status "INFO" "Stopping Docker containers..."
docker-compose down

# Remove any orphaned containers
print_status "INFO" "Removing any orphaned containers..."
docker-compose down --remove-orphans

# Remove workshop images for fresh rebuild
print_status "INFO" "Removing workshop Docker images..."
docker images | grep rag-workshop | awk '{print $3}' | xargs docker rmi -f 2>/dev/null || true

# Optional: Remove Docker volumes (uncomment if you want to clean all data)
# print_status "INFO" "Removing Docker volumes..."
# docker-compose down --volumes

# Check if containers are stopped
print_status "INFO" "Checking container status..."
RUNNING_CONTAINERS=$(docker-compose ps --services --filter "status=running" 2>/dev/null)

if [ -z "$RUNNING_CONTAINERS" ]; then
    print_status "SUCCESS" "All containers stopped successfully"
else
    print_status "WARNING" "Some containers may still be running:"
    docker-compose ps
fi

# Optional: Clean up Docker system (uncomment for deep clean)
# print_status "INFO" "Cleaning up Docker system..."
# docker system prune -f

print_status "INFO" "Workshop cleanup complete!"
print_status "INFO" "To start fresh, run: ./workshop.sh"

echo "=================================================="
