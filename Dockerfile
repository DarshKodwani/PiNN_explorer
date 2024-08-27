# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app code into the container
COPY . .

# Expose the port FastAPI is running on
EXPOSE 8000

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "src.azure_infra.create_cavity_lid_api:app", "--host", "0.0.0.0", "--port", "8000"]