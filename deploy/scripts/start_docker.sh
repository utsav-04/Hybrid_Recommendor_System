#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 073987696099.dkr.ecr.ap-south-1.amazonaws.com

echo "Pulling Docker image..."
docker pull 073987696099.dkr.ecr.ap-south-1.amazonaws.com/tasun-recommendor-system:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=tasun-recommendor-system)" ]; then
    echo "Stopping existing container..."
    docker stop hybrid_recsys
fi

if [ "$(docker ps -aq -f name=tasun-recommendor-system)" ]; then
    echo "Removing existing container..."
    docker rm hybrid_recsys
fi

echo "Starting new container..."
docker run -d -p 80:8000 --name tasun-recommendor-system 073987696099.dkr.ecr.ap-south-1.amazonaws.com/tasun-recommendor-system:latest
                                                        
echo "Container started successfully."