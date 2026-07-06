# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for OpenCV and PyTorch)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Command to run the application using Gunicorn for production
RUN pip install gunicorn
# Create a non-root user with uid 1000 (required by Hugging Face)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Change ownership of the app directory to the new user
RUN chown -R user:user /app

# Switch to the non-root user
USER user

# Run gunicorn bound to the dynamic PORT or default 7860
CMD gunicorn --bind 0.0.0.0:${PORT:-7860} --error-logfile - --access-logfile - app:app --timeout 120 --workers 1 --threads 2
