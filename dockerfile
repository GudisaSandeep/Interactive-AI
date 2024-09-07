# Use the official Python image from DockerHub
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt .

# Install the required Python packages
RUN python -m venv --copies /opt/venv && . /opt/venv/bin/activate && pip install -r requirements.txt

# Copy the entire project folder into the container
COPY . .

# Expose the port Flask will run on
EXPOSE 5000

# Define the command to run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
