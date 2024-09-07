# Use official Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code to the container
COPY . .

# Expose the port on which the app runs
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
