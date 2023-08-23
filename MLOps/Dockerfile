# Using official Anaconda base image
FROM continuumio/anaconda3:latest

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Updating apt-get and installing some dependencies that are needed
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Setting up work directory
WORKDIR /app

# Copying requirements.txt to the container
COPY ./requirements.txt .

# Installing Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copying the rest of the code and model into the container
COPY . .

# Running Flask API on the container start
CMD [ "python", "inference.py" ]
