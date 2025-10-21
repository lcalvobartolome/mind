import os
import json
from flask import Flask, request, jsonify

app = Flask(__name__)
PORT = 5001 


@app.route('/process/preprocessing', methods=['POST'])
def handle_preprocessing():
    """Ruta que recibe la petición del Frontend y ejecuta el proceso pesado."""
    try:
        data = request.get_json()
        dataset_name = data.get('dataset_name')
        email = data.get('email')
        segmenter_data = data.get('segmenter_data')
        translator_data = data.get('translator_data')
        preparer_data = data.get('preparer_data')
        
        if not dataset_name or not email or not segmenter_data or not translator_data or not preparer_data:
            return jsonify({"status": "error", "message": "Not found at least one of the arguments arguments"}), 400

        print(f"WORKER: Preprocessing {dataset_name}...")

        # Teniendo la info
        results = "llego"

        print(f"WORKER: Preprocesamiento completado exitosamente.")
        
        # 3. Devolver la respuesta al Frontend
        return jsonify({
            "status": "success",
            "message": f"Preprocesamiento completado para {dataset_name}.",
            "results": results
        }), 200

    except Exception as e:
        print(f"WORKER ERROR: {e}")
        return jsonify({"status": "error", "message": f"Fallo interno del Worker: {str(e)}", "next_step_available": False}), 500

if __name__ == '__main__':
    from dataset import datasets_bp
    from preprocessing import preprocessing_bp
    
    app.register_blueprint(datasets_bp, url_prefix='/')
    app.register_blueprint(preprocessing_bp, url_prefix='/')
    
    app.run(host='0.0.0.0', port=PORT)
