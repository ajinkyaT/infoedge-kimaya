# Use the official Python image from the Docker Hub that matches the version requirement
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files to the container
COPY pyproject.toml poetry.lock /app/

# Install Poetry
RUN pip install poetry

# Install the dependencies
RUN poetry install --no-root

# Copy the rest of the application code to the container
COPY . /app

# Expose the port that Streamlit will run on
EXPOSE 8501

# Set environment variable for the port
ENV PORT 8080

# Set the entry command to run the Streamlit app using Poetry
CMD ["poetry", "run", "streamlit", "run", "streamlit_app.py", "--server.port", "8080"]