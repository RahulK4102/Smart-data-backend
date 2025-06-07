# Base image with Python installed
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy your entire project into the container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Create required folders (if not already in repo)
RUN mkdir -p project_root/data project_root/outputs project_root/mapped

# Expose the port Gunicorn will run on
EXPOSE 8000

# Start the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "main:app"]

