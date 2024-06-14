FROM python:3.9

# Set up working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install additional dependencies
RUN pip install -r requirements.txt

# Expose Streamlit port
EXPOSE 2092

# Command to run the  app
CMD ["python3", "app2.py"]