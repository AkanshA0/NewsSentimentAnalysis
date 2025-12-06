#!/bin/bash

# Startup script for EC2 instance
# This runs automatically when instance starts

echo "Starting Stock Prediction Application..."

# Navigate to app directory
cd /home/ubuntu/NewsSentiment

# Pull latest changes (if using git)
# git pull origin main

# Start Docker containers
docker-compose up -d

echo "Application started successfully!"
echo "Access at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8501"
