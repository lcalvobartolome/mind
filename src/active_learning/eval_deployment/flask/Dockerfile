FROM python:3.10-slim

WORKDIR /app

# Copy the requirements file
COPY requirements.txt requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ../../../../eval_deployment/flask /app/flask
COPY ../../../../__init__.py /app/__init__.py
COPY ../../../../blade.py /app/blade.py
COPY ../../../../doc_selector.py /app/doc_selector.py
COPY ../../../../test.ipynb /app/test.ipynb
COPY ../../../../words.txt /app/words.txt

# Expose the port the app runs on
EXPOSE 5000

# Command to run the app
CMD ["python", "flask/app.py"]