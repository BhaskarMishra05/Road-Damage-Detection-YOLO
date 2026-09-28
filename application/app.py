import os
import sys
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

# ---------------------------------------------------------
# Path Configurations
# ---------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import updated prediction pipeline
from src.pipeline.prediction_pipeline import Prediction, PredictionConfig

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


def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set


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

            # ---------------------------------------------------------
            # 1. Image Classification Mode
            # ---------------------------------------------------------
            if media_type == 'image':
                res = predictor.image_classifier(input_path, generate_pdf=True)
                
                # Image save path relative to static/
                annotated_img_rel_path = f"output/{base_name}_op/{filename}"
                pdf_filename = f"{base_name}_report.pdf"

                return render_template(
                    'result.html',
                    media_type='image',
                    result_image=annotated_img_rel_path,
                    pdf_file=pdf_filename,
                    counts=res['counts'],
                    damage_score=res['severity_score'],
                    priority=res['priority'],
                    remark=res['remark']
                )

            # ---------------------------------------------------------
            # 2. Video Classification Mode (No GPS)
            # ---------------------------------------------------------
            elif media_type == 'video':
                res = predictor.video_classifier(input_path, display_preview=False, generate_pdf=True)

                video_filename = f"{base_name}_detected.mp4"
                pdf_filename = f"{base_name}_report.pdf"

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

            # ---------------------------------------------------------
            # 3. Video + GPS Telemetry Mode (500m Chainage Sets)
            # ---------------------------------------------------------
            elif media_type == 'video_gps':
                if 'gps_file' not in request.files:
                    return render_template('index.html', error="GPS CSV file is required for Video + GPS mode.")

                gps_file = request.files['gps_file']

                if gps_file.filename == '' or not allowed_file(gps_file.filename, ALLOWED_GPS_EXTENSIONS):
                    return render_template('index.html', error="Valid GPS .csv file is required.")

                gps_filename = secure_filename(gps_file.filename)
                gps_path = os.path.join(app.config['UPLOAD_FOLDER'], gps_filename)
                gps_file.save(gps_path)

                res = predictor.video_gps_classifier(input_path, gps_path, generate_pdf=True)

                video_filename = f"{base_name}_gps_detected.mp4"
                pdf_filename = f"{base_name}_gps_report.pdf"

                return render_template(
                    'result.html',
                    media_type='video_gps',
                    video_file=video_filename,
                    pdf_file=pdf_filename,
                    total_distance_km=res['total_distance_km'],
                    total_segments=res['total_segments'],
                    segments=res['segments']
                )
        else:
            return render_template('index.html', error="Invalid file format.")

    return render_template('index.html')


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)