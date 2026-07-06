# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for OpenCV and PyTorch)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 5000 (Render uses PORT env variable, we'll bind to 0.0.0.0:PORT)
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Command to run the application using Gunicorn for production
# Install gunicorn just in case (or add to requirements.txt)
RUN pip install gunicorn

# Run gunicorn bound to the dynamic PORT provided by Render or default 5000
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} app:app --timeout 120 --workers 1 --threads 2
