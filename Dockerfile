# Use the official Python base image
FROM python:3.10-slim

# Install necessary build tools and Rust
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && . "$HOME/.cargo/env"

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script (converted from Jupyter Notebook) into the container
COPY assessment.py .

# (Optional) If you have a data directory, copy it too
COPY data/ ./data/

# (Optional) If you have an output directory for storing models or results, create it
RUN mkdir -p output

# Set the default command to run your Python script
CMD ["python", "./assessment.py"]
