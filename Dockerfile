# Use an official Python image as the base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the environment.yml file to the working directory
COPY ./environment.yml .

# Install dependencies
RUN conda env create -f environment.yml

# Make sure the environment is activated:
SHELL ["conda", "run", "-n", "llama-3.1", "/bin/bash", "-c"]

# Copy the project files to the working directory
COPY . .

# Set the default command to run your app
# Replace `python app.py` with your startup command
CMD ["python", "app.py"]
