# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy backend code
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose Flask port
EXPOSE 5050

# Set environment variables (optional)
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Start the Flask app
CMD ["python", "app.py"]
