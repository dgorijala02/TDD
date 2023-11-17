# Use an official Python runtime as a base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the local source code to the container
COPY . .

# Install dependencies
RUN pip install pytest

# Define the command to run your application (tests in this case)
CMD ["pytest", "test_sparse_recommender.py"]

