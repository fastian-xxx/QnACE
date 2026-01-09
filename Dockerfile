# Dockerfile for Q&ACE Backend
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY integrated_system/requirements.txt requirements.txt
COPY interview_emotion_detection/requirements.txt requirements_emotion.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_emotion.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8001

# Set environment variables
ENV PYTHONPATH=/app/integrated_system:/app/interview_emotion_detection/src

# Run the API server
CMD ["python", "integrated_system/api/main.py"]