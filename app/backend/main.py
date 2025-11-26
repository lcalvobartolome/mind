import sys
import dotenv

from flask import Flask, request


app = Flask(__name__)
PORT = 5001
dotenv.load_dotenv()


@app.before_request
def cancel_active_pipeline_if_needed():
    if request.path == "/log_detection":
        return

    email = request.args.get("email") or request.args.get("user_id")
    if not email and request.is_json:
        data = request.get_json()
        email = data.get("email") or data.get("user_id")
    if not email:
        return
    

    from detection import active_processes, lock
    with lock:
        if email in active_processes:
            proc_info = active_processes[email]
            print(f"Cancelling active pipeline for {email} due to request {request.path}", file=sys.__stdout__)
            
            proc_info["cancel_event"].set()
            if proc_info["process"].is_alive():
                proc_info["process"].terminate()
                proc_info["process"].join()
            
            del active_processes[email]
            print(active_processes)

@app.route('/cancel_detection', methods=['GET'])
def route_cancel_detection():
    return {}, 200

if __name__ == '__main__':
    from dataset import datasets_bp
    from detection import detection_bp
    from preprocessing import preprocessing_bp
    
    app.register_blueprint(datasets_bp, url_prefix='/')
    app.register_blueprint(detection_bp, url_prefix='/')
    app.register_blueprint(preprocessing_bp, url_prefix='/')
    
    app.run(host='0.0.0.0', port=PORT)
