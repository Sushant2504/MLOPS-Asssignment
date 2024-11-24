# Use a lightweight Python image with Streamlit pre-installed
FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt ./

# Install dependencies
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Expose the port used by Streamlit (default 8501)
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "main.py"]  # Replace "main.py" 