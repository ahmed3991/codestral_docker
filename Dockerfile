# Base image with CUDA support
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git wget unzip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python libraries
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    kaggle \
    transformers \
    bitsandbytes \
    accelerate \
    langchain \
    langchain_community \
    flask

# Set working directory
WORKDIR /app

RUN mkdir -p /root/.kaggle

# Copy Kaggle API token (make sure it's in the same directory as the Dockerfile)
COPY kaggle.json /root/.kaggle/kaggle.json

# Ensure proper permissions for the Kaggle API token
RUN chmod 600 /root/.kaggle/kaggle.json

RUN mkdir -p /app/model

# Download the model from Kaggle
RUN kaggle models instances versions download ahmed3991/icodestral-22b-v0-1-hf-model/pyTorch/icodestral-22b-v0-1-hf-model-v1/1 -p /app/model && \
    unzip /app/model/icodestral-22b-v0-1-hf-model.zip -d /app/model && \
    rm /app/model/icodestral-22b-v0-1-hf-model.zip && \
    rm /root/.kaggle/kaggle.json

RUN ls /app/model

# Copy app.py
COPY app.py /app/app.py

# Expose Flask app port
EXPOSE 5000

# Command to run Flask app
CMD ["python3", "app.py"]
