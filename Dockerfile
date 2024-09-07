# Use the official Python base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# Add the `--verbose` flag to get detailed error messages
RUN pip install --no-cache-dir --verbose -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port your app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
