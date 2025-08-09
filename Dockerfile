# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY front-end/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pull DVC data
RUN dvc pull

# Copy the rest of the application's code
COPY front-end/front_end_app.py .

# Expose the port on which the app will run
EXPOSE 8050

# Define the command to run the application
CMD ["python", "front_end_app.py"]