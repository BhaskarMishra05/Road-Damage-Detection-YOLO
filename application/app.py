import os
import sys
import uuid
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# ---------------------------------------------------------
# Path Configurations
# ---------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import updated prediction pipeline and database functions
from src.pipeline.prediction_pipeline import Prediction, PredictionConfig
from src.database.db import (
    init_db, create_run, add_detections, update_run_outputs, 
    get_recent_runs, get_severity_score
)

# Initialize database tables on application startup
init_db()

# Set template and static folders explicitly
TEMPLATE_DIR = os.path.join(APP_DIR, 'templates')
STATIC_DIR = os.path.join(APP_DIR, 'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# Upload and Output paths inside application/static
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(STATIC_DIR, 'output')
ALLOWED_MEDIA_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_GPS_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize Prediction Pipeline
config = PredictionConfig(output_dir=OUTPUT_FOLDER)
predictor = Prediction(config=config)

# In-memory background job store for asynchronous video/GPS processing (Review Paper Section XI)
JOBS = {}


def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set


# ---------------------------------------------------------
# Background Worker for Asynchronous Video Analysis
# ---------------------------------------------------------
def run_background_job(job_id, media_type, input_path, gps_path, run_id, base_name, filename):
    try:
        JOBS[job_id] = {"status": "processing", "progress": "Running model pipeline..."}
        
        if media_type == 'video':
            res = predictor.video_classifier(input_path, display_preview=False, generate_pdf=True)
            output_rel_path = f"output/{base_name}_detected.mp4"
            report_rel_path = f"output/{base_name}_report.pdf"
        elif media_type == 'video_gps':
            res = predictor.video_gps_classifier(input_path, gps_path, generate_pdf=True)
            output_rel_path = f"output/{base_name}_gps_detected.mp4"
            report_rel_path = f"output/{base_name}_gps_report.pdf"
        else:
            return

        if 'detections' in res and res['detections']:
            add_detections(run_id, res['detections'])
        update_run_outputs(run_id, output_rel_path, report_rel_path)

        JOBS[job_id] = {
            "status": "completed",
            "media_type": media_type,
            "result": res,
            "output_path": output_rel_path,
            "report_path": report_rel_path
        }
    except Exception as e:
        JOBS[job_id] = {"status": "failed", "error": str(e)}


# ---------------------------------------------------------
# Web Routes (Server-Rendered Dashboard)
# ---------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        media_type = request.form.get('media_type')

        if 'file' not in request.files:
            return render_template('index.html', error="No primary media file uploaded.")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected media file.")

        if file and allowed_file(file.filename, ALLOWED_MEDIA_EXTENSIONS):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)

            base_name = os.path.splitext(filename)[0]
            relative_input_path = f"uploads/{filename}"
            run_id = create_run(media_type, relative_input_path)

            # Image Mode processes synchronously (instant)
            if media_type == 'image':
                res = predictor.image_classifier(input_path, generate_pdf=True)
                output_rel_path = f"output/{base_name}_op/{filename}"
                report_rel_path = f"output/{base_name}_report.pdf"

                if 'detections' in res and res['detections']:
                    add_detections(run_id, res['detections'])
                update_run_outputs(run_id, output_rel_path, report_rel_path)

                return render_template(
                    'result.html',
                    media_type='image',
                    result_image=output_rel_path,
                    pdf_file=f"{base_name}_report.pdf",
                    counts=res['counts'],
                    damage_score=res['severity_score'],
                    priority=res['priority'],
                    remark=res['remark']
                )

            # Video Modes offloaded to background job handling per review paper spec
            elif media_type in ['video', 'video_gps']:
                gps_path = None
                if media_type == 'video_gps':
                    if 'gps_file' not in request.files:
                        return render_template('index.html', error="GPS CSV file is required for Video + GPS mode.")
                    gps_file = request.files['gps_file']
                    if gps_file.filename == '' or not allowed_file(gps_file.filename, ALLOWED_GPS_EXTENSIONS):
                        return render_template('index.html', error="Valid GPS .csv file is required.")
                    gps_filename = secure_filename(gps_file.filename)
                    gps_path = os.path.join(app.config['UPLOAD_FOLDER'], gps_filename)
                    gps_file.save(gps_path)

                job_id = str(uuid.uuid4())
                threading.Thread(
                    target=run_background_job,
                    args=(job_id, media_type, input_path, gps_path, run_id, base_name, filename)
                ).start()

                return render_template('job_status.html', job_id=job_id, media_type=media_type)

        else:
            return render_template('index.html', error="Invalid file format.")

    return render_template('index.html')


@app.route('/result/<job_id>')
def job_result(job_id):
    """Renders result.html once the background video or video+GPS job finishes."""
    job = JOBS.get(job_id)
    if not job or job.get("status") != "completed":
        return render_template('index.html', error="Job not found or still processing.")

    res = job["result"]
    media_type = job["media_type"]
    video_filename = os.path.basename(job["output_path"])
    pdf_filename = os.path.basename(job["report_path"])

    if media_type == 'video':
        return render_template(
            'result.html',
            media_type='video',
            video_file=video_filename,
            pdf_file=pdf_filename,
            counts=res['counts'],
            damage_score=res['severity_score'],
            priority=res['priority'],
            remark=res['remark']
        )
    elif media_type == 'video_gps':
        return render_template(
            'result.html',
            media_type='video_gps',
            video_file=video_filename,
            pdf_file=pdf_filename,
            total_distance_km=res['total_distance_km'],
            total_segments=res['total_segments'],
            segments=res['segments']
        )


@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


# ---------------------------------------------------------
# REST API Endpoints (As Specified in Review Paper Table XII)
# ---------------------------------------------------------
@app.route('/api/video/analyze', methods=['POST'])
def api_video_analyze():
    """API endpoint for video analysis returning a job ID for polling."""
    if 'file' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename, ALLOWED_MEDIA_EXTENSIONS):
        return jsonify({"error": "Invalid or missing video file"}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    base_name = os.path.splitext(filename)[0]
    run_id = create_run('video', f"uploads/{filename}")
    job_id = str(uuid.uuid4())

    threading.Thread(
        target=run_background_job,
        args=(job_id, 'video', input_path, None, run_id, base_name, filename)
    ).start()

    return jsonify({"job_id": job_id, "run_id": run_id, "status": "processing"}), 202


@app.route('/api/video-gps/analyze', methods=['POST'])
def api_video_gps_analyze():
    """API endpoint for video + GPS CSV analysis."""
    if 'file' not in request.files or 'gps_file' not in request.files:
        return jsonify({"error": "Both video and GPS CSV files are required"}), 400

    file = request.files['file']
    gps_file = request.files['gps_file']

    if not allowed_file(file.filename, ALLOWED_MEDIA_EXTENSIONS) or not allowed_file(gps_file.filename, ALLOWED_GPS_EXTENSIONS):
        return jsonify({"error": "Invalid file formats"}), 400

    filename = secure_filename(file.filename)
    gps_filename = secure_filename(gps_file.filename)

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    gps_path = os.path.join(app.config['UPLOAD_FOLDER'], gps_filename)

    file.save(input_path)
    gps_file.save(gps_path)

    base_name = os.path.splitext(filename)[0]
    run_id = create_run('video_gps', f"uploads/{filename}")
    job_id = str(uuid.uuid4())

    threading.Thread(
        target=run_background_job,
        args=(job_id, 'video_gps', input_path, gps_path, run_id, base_name, filename)
    ).start()

    return jsonify({"job_id": job_id, "run_id": run_id, "status": "processing"}), 202


@app.route('/api/jobs/<job_id>', methods=['GET'])
def api_job_status(job_id):
    """Poll progress of a running video or GPS analysis job."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job ID not found"}), 404
    return jsonify(job), 200


@app.route('/api/runs', methods=['GET'])
def api_list_runs():
    """List recent audit runs stored in MySQL."""
    try:
        runs = get_recent_runs(limit=10)
        return jsonify({"runs": runs}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/runs/<int:run_id>', methods=['GET'])
def api_get_run_details(run_id):
    """Retrieve details, counts, and severity scores for a specific run."""
    try:
        score = get_severity_score(run_id)
        return jsonify({"run_id": run_id, "severity_score": score}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)