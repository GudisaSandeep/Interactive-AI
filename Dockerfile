# Use an official Python runtime as a parent image
FROM python:3.9
# Set the working directory in the container
WORKDIR /app
# Install system dependencies required for pyaudio
RUN apt-get update && \
    apt-get install -y \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*
# Upgrade pip
RUN pip install --upgrade pip
# Copy the current directory contents into the container at /app
COPY . /app
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Make port 7860 available to the world outside this container
EXPOSE 7860
# Define environment variable
ENV NAME World
# Run app.py when the container launches
CMD ["python", "app.py"]
