# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Expose the port on which the app will run
EXPOSE 8000

# Run the API with Uvicorn, and make it accessible from outside the container
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]