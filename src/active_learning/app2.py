"""
Main application entry point
"""
import logging
from flask import Flask, render_template, jsonify, request
from flask_restx import Api
from threading import Thread
import time
import requests
from pyfiglet import figlet_format
from termcolor import cprint

# Import the namespace
from apis.namespace import api as active_learning_api

# Create Flask app
app = Flask(__name__)
# Deactivate the default mask parameter
app.config["RESTX_MASK_SWAGGER"] = False

# Initialize API and add the namespace
api = Api(
    title="EWB's Topic Modeling API",
    version='1.0',
    description='whatever',
)
api.add_namespace(active_learning_api, path='/test')
api.init_app(app)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Endpoint to fetch a new document
@app.route('/get_new_document', methods=['POST'])
def get_new_document():
    idx = 0  # Replace with actual logic to get document index
    logger.info("Calling getDocumentToLabel")
    response = requests.post('http://kumo01.tsc.uc3m.es:2095/test/getDocumentToLabel/', data={'idx': idx})
    if response.status_code != 200:
        logger.error(f"Error calling getDocumentToLabel: {response.json()}")
        return jsonify({"error": "Failed to fetch document"}), 500

    document = response.json()
    logger.info(f"Fetched document: {document}")
    return jsonify(document)

# Endpoint to submit annotation
@app.route('/submit_annotation', methods=['POST'])
def submit_annotation():
    label = request.form['label']
    logger.info("Calling LabelDocument")
    response = requests.post('http://kumo01.tsc.uc3m.es:2095/test/LabelDocument/', data={'label': label})
    if response.status_code != 200:
        logger.error(f"Error calling LabelDocument: {response.json()}")
        return jsonify({"error": "Failed to label document"}), 500

    logger.info("Document labeled successfully")
    return jsonify({"message": "Document labeled successfully"}), 200

# Endpoint to start the task
@app.route('/startAnnotationTask', methods=['POST'])
def start_annotation_task():
    logger.info("Starting annotation task")
    thread = Thread(target=run_annotation_task)
    thread.start()
    return jsonify({"message": "Annotation task started"}), 200

# Background task to run the annotation task (optional, for background processing)
def run_annotation_task(duration=3600):
    start_time = time.time()
    while time.time() - start_time < duration:
        idx = 0  # Replace with actual logic to get document index
        logger.info("Calling getDocumentToLabel")
        response = requests.post('http://kumo01.tsc.uc3m.es:2095/test/getDocumentToLabel/', data={'idx': idx})
        if response.status_code != 200:
            logger.error(f"Error calling getDocumentToLabel: {response.json()}")
            break

        document = response.json()
        logger.info(f"Fetched document: {document}")
        time.sleep(5)  # Adjust sleep time as needed

# Serve the frontend
@app.route('/index.html')
def index():
    return render_template('index.html')

# Serve the Swagger UI at /swaggerui.html
@app.route('/swaggerui.html')
def serve_swagger():
    return render_template('swaggerui.html')

# Redirect to Swagger UI
@app.route('/api/docs')
def swagger_ui():
    return redirect('/swaggerui.html')

if __name__ == '__main__':
    cprint(figlet_format("ROISE", font='big'), 'blue', attrs=['bold'])
    print('\n')
    
    logger.info("Starting the Flask app")
    
    try:
        app.run(host='0.0.0.0', port=2095, debug=True)
        # from waitress import serve
        # serve(app, host="0.0.0.0", port=2092)
    except Exception as e:
        logger.error("Error occurred while running the Flask app", exc_info=True)