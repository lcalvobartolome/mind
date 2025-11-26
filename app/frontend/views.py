import os
import json
import queue
import dotenv
import requests

from functools import wraps
from detection import getTMDatasets, getTMkeys, analyseContradiction, get_result_mind
from flask import Blueprint, Response, render_template, request, flash, jsonify, session, url_for


views = Blueprint('views', __name__)
dotenv.load_dotenv()

MIND_WORKER_URL = os.environ.get('MIND_WORKER_URL')
AUTH_API_URL = f"{os.environ.get('AUTH_API_URL', 'http://auth:5002/')}/auth"
log_queue = queue.Queue() 


def login_required_custom(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "warning")
            return render_template("home.html")
        return f(*args, **kwargs)
    return decorated_function

@views.route('/')
def home():
    user_id = session.get('user_id')
    try:
        requests.get(f"{MIND_WORKER_URL}/cancel_detection", params={"email": user_id})
    except:
        pass
    return render_template("home.html", user_id=user_id)

@views.route('/detectionAllResults')
@login_required_custom
def detection_AllResults_page():
    user_id = session.get('user_id')
    try:
        response = requests.get(f"{MIND_WORKER_URL}/datasets", params={"email": user_id})
        if response.status_code == 200:
            data = response.json()
            datasets = data.get("datasets", [])
            names = data.get("names", [])
            shapes = data.get("shapes", [])
            stages = data.get("stages", [])
        else:
            flash(f"Error loading datasets: {response.text}", "danger")
            datasets, names, shapes, stages = [], [], [], []
    except requests.exceptions.RequestException:
        flash("Backend service unavailable.", "danger")
        datasets, names, shapes, stages = [], [], [], []

    return render_template("detection_results.html", user_id=user_id, datasets=datasets, names=names, shape=shapes, stages=stages, zip=zip)

@views.route('/dataset_selection', methods=['POST'])
@login_required_custom
def dataset_selection():
    data = request.get_json()
    dataset = data.get('dataset')

    if not dataset:
        flash('No dataset provided', 'danger')
        return jsonify({'error': 'No dataset provided'}), 400

    print("Received dataset:", dataset)
    session['dataset'] = dataset
    
    return jsonify({'message': 'Dataset received', 'dataset': dataset})

@views.route('/mode_selection', methods=['GET', 'POST'])
@login_required_custom
def mode_selection():
    data = request.get_json()
    mode = data.get('instruction')
    print("Received mode:", mode)

    user_id = session['user_id']

    if not mode:
        flash('No mode provided', 'danger')
        return jsonify({'error': 'No mode provided'}), 400

    if request.method == 'POST':
        if mode == "Analyze contradictions":
            topics = data.get('topics')
            config = data.get('config')
            
            # Call backend
            result = analyseContradiction(user_id, session.get('TM'), topics, data.get('sample_size'), config)
            return result
        
        else:
            flash('Invalid mode selected', 'danger')
            return jsonify({'error': 'Invalid mode selected'}), 400
    else:
        flash(f"Select a dataset before exploring!", "warning")
        return jsonify({'error': f'MIND is not initialized.'})
    
@views.route("/log_detection", methods=["POST"])
def receive_log():
    data = request.get_json()
    log_line = data.get("log")
    if log_line:
        log_queue.put(log_line)
    return {"status": "ok"}

@views.route("/stream_detection")
@login_required_custom
def stream():
    def event_stream():
        while True:
            try:
                line = log_queue.get(timeout=1)
                yield f"data: {json.dumps({'log': line})}\n\n"
            except queue.Empty:
                continue
    return Response(event_stream(), mimetype="text/event-stream")
    
@views.route('/detection_results')
@login_required_custom
def detection_results_page():
    result_mind = None
    result_columns = None
    
    try:
        user_id = session['user_id']
        TM = request.args.get('TM')
        topics = request.args.get('topics')

        result = get_result_mind(user_id, TM, topics)
        if result is None:
            flash('Error in Backend', 'warning')
            return result
        
        result_mind = result.get('result_mind')
        result_columns = result.get('result_columns')
        columns_json = result.get('columns_json')
        non_orderable_indices = result.get('non_orderable_indices')

        return render_template("detection.html", user_id=user_id, status="completed", result_mind=result_mind, result_columns=result_columns, columns_json=columns_json, non_orderable_indices=non_orderable_indices)
    
    except Exception as e:
        print(e)
        return render_template("detection.html", user_id=user_id, status="completed", result_mind=result_mind, result_columns=result_columns, columns_json=columns_json, non_orderable_indices=non_orderable_indices)

@views.route('/update_results', methods=['POST'])
@login_required_custom
def update_mind_results():
    try:
        user_id = session['user_id']

        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        TM = request.form.get('TM')
        topics = request.form.get('topics')

        if not TM or not topics:
            return jsonify({"message": "Missing fields"}), 400

        files = {"file": (file.filename, file.stream, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        data = {
            "TM": TM,
            "topics": topics,
            "email": user_id
        }

        response = requests.post(f"{MIND_WORKER_URL}/detection/update_results", files=files, data=data)

        if response.status_code != 200:
            return jsonify({"message": "Error from backend"}), 500

        return jsonify({"message": "All changes has been updated."}), 200
    
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@views.route('/detection', methods=['GET', 'POST'])
@login_required_custom
def detection_page():
    user_id = session.get('user_id')

    status = "idle"
    mind_info = {}
    dataset_detection = {}
    topic_keys = {}

    if request.method == 'GET':
        dataset_detection = getTMDatasets(user_id)

    elif request.method == 'POST':
        try:
            data = request.get_json()
            print(f'Found: {data}')
            data = json.loads(data)
            session["TM"] = data['topic_model']

        except Exception as e:
            flash("Couldn't get correctly data or communicate to the backend.", "danger")
            dataset_detection = getTMDatasets(user_id)

        if data:
            # Get the topic keys from that model
            topic_keys = getTMkeys(user_id, data)

    return render_template("detection.html", user_id=user_id, status=status, dataset_detection=dataset_detection, topic_keys=topic_keys, mind_info=mind_info)

@views.route('/detection_topickeys', methods=['POST'])
@login_required_custom
def detection_page_topickeys_post():
    user_id = session.get('user_id')
    
    status = "idle"
    mind_info = {}
    session['mind_info'] = mind_info
    dataset_detection = ""
    session['dataset_detection'] = None
    topic_keys = {}

    try:
        data = request.get_json()
        print(f'Found: {data}')
        data = json.loads(data)
        session["TM"] = data['topic_model']

    except Exception as e:
        flash("Couldn't get correctly data or communicate to the backend.", "danger")
        dataset_detection = getTMDatasets(user_id)
        session['dataset_detection'] = dataset_detection

    if data:
        # Get the topic keys from that model
        topic_keys = getTMkeys(user_id, data)
        session['topic_keys'] = topic_keys

    return jsonify({
        'redirect': url_for('views.detection_page_topickeys_get',
                            status=status)
    })

@views.route('/detection_topickeys', methods=['GET'])
@login_required_custom
def detection_page_topickeys_get():
    user_id = session.get('user_id')
    status = request.args.get('status')
    dataset_detection = session.get('dataset_detection')
    topic_keys = session.get('topic_keys')
    mind_info = session.get('mind_info')

    return render_template(
        "detection.html",
        user_id=user_id,
        status=status,
        dataset_detection=dataset_detection,
        topic_keys=topic_keys,
        mind_info=mind_info
    )
