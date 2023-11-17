# Use an official Node.js runtime as a base image
FROM node:14

# Set the working directory in the container
WORKDIR /

# Copy package.json and package-lock.json to the working directory
COPY sparse_recommender.py .
COPY test_sparse_recommender.py .

# Copy the local source code to the container
COPY . .

# Define the command to run your application
CMD ["pytest", "test_sparse_recommender.py"]
