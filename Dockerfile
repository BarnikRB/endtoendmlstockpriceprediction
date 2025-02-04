FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run api.py when the container launches
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]