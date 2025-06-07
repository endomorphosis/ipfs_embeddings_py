# Use the NVIDIA CUDA base image for Ubuntu
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    netcat \
    sudo \
    wget \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /home/devel/laion-embeddings

# Copy the current directory contents into the container
COPY . .

# Install Python dependencies
RUN pip3 install  --no-cache-dir -r requirements.txt

# Set the entrypoint
ENTRYPOINT ["python3", "-m", "fastapi", "run" , "main.py"]