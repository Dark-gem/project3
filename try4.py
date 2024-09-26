import os
import logging
import threading
import time

from flask import Flask, render_template, Response, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import requests
from datetime import datetime
import pandas as pd
import face_recognition
import torch
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch import InceptionResnetV1, MTCNN


# Initialize Flask application
app = Flask(__name__)

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///C:/Users/DELL/PycharmProjects/project3/instance/employees.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_unique_secret_key_here')

db = SQLAlchemy(app)

# Logging configuration
logging.basicConfig(level=logging.DEBUG)

# Models
class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    position = db.Column(db.String(50))
    phone = db.Column(db.String(20))
    email = db.Column(db.String(100))
    job_type = db.Column(db.String(20))
    checkin_time = db.Column(db.String(20))
    checkout_time = db.Column(db.String(20))
    photo_filename = db.Column(db.String(100))
    face_encoding = db.Column(db.LargeBinary)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_name = db.Column(db.String(100))
    year = db.Column(db.Integer)
    month = db.Column(db.Integer)
    day = db.Column(db.Integer)
    checkin_time = db.Column(db.String(20))
    checkout_time = db.Column(db.String(20))

# Initialize face recognition model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

def record_attendance(employee_name):
    now = datetime.now()
    new_record = Attendance(
        employee_name=employee_name,
        year=now.year,
        month=now.month,
        day=now.day,
        checkin_time=now.strftime('%H:%M:%S'),
        checkout_time=None  # Update when checking out
    )
    db.session.add(new_record)
    db.session.commit()

def preprocess_image(image):
    image = cv2.resize(image, (160, 160))
    image = image.astype(np.float32) / 255.0
    return torch.tensor(image).unsqueeze(0)

def predict_face(image):
    image = preprocess_image(image)
    embedding = facenet_model(image).detach().numpy()[0]
    return embedding

def detect_faces(frame):
    # Detect faces in the frame using MTCNN
    boxes, _ = mtcnn.detect(frame)
    return boxes


def recognize_faces(frame, boxes):
    # Preprocess and recognize faces in the detected boxes
    recognized_faces = []
    if boxes is not None:
        faces = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))
            face = face.astype(np.float32) / 255.0
            face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0)  # Reorder dimensions for facenet
            faces.append(face)

        if faces:
            faces = torch.cat(faces)
            embeddings = facenet_model(faces).detach().cpu().numpy()

            # Compare embeddings with known faces
            for embedding in embeddings:
                name, similarity = match_face(embedding)
                recognized_faces.append((name, similarity))

    return recognized_faces


def match_face(embedding):
    # Compare the embedding with known face embeddings
    known_face_encodings, known_face_names = load_face_encodings_from_db()
    min_distance = float("inf")
    best_match = "Unknown"

    for known_embedding, name in zip(known_face_encodings, known_face_names):
        distance = np.linalg.norm(known_embedding - embedding)
        if distance < min_distance:
            min_distance = distance
            best_match = name

    # Define a threshold for recognition
    if min_distance < 0.8:
        return best_match, min_distance
    else:
        return "Unknown", None


# def compare_embeddings(embedding1, embedding2):
#     dot_product = np.dot(embedding1, embedding2)
#     norm1 = np.linalg.norm(embedding1)
#     norm2 = np.linalg.norm(embedding2)
#     return dot_product / (norm1 * norm2)

def load_employee_data():
    employees = Employee.query.all()
    return [(emp.id, emp.name, os.path.join(app.config['UPLOAD_FOLDER'], emp.photo_filename)) for emp in employees]

def save_employee_data(name, position, phone, email, job_type, photo, checkin_time, checkout_time):
    photo_filename = secure_filename(photo.filename) if photo else None
    face_encoding = None
    if photo:
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], photo_filename)
        photo.save(photo_path)

        # Generate face encoding
        image = face_recognition.load_image_file(photo_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            face_encoding = encodings[0].tobytes()

    new_employee = Employee(
        name=name,
        position=position,
        phone=phone,
        email=email,
        job_type=job_type,
        checkin_time=checkin_time,
        checkout_time=checkout_time,
        photo_filename=photo_filename,
        face_encoding=face_encoding
    )
    db.session.add(new_employee)
    db.session.commit()

def update_checkin_time(employee_id):
    employee = Employee.query.get(employee_id)
    if employee:
        today = datetime.today().date()
        employee.checkin_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        db.session.commit()

def get_face_embedding(image):
    face_locations = face_recognition.face_locations(image, model="hog")
    encodings = face_recognition.face_encodings(image, face_locations)
    app.logger.debug(f"Face locations: {face_locations}")
    app.logger.debug(f"Encodings: {encodings}")
    return encodings, face_locations

def recognize_face(face_encoding, known_face_encodings, known_face_names, threshold=0.2):
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    app.logger.debug(f"Face distances: {face_distances}")

    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=threshold)
    app.logger.debug(f"Matches: {matches}")

    if len(face_distances) > 0:
        best_match_index = np.argmin(face_distances)
        app.logger.debug(f"Best match index: {best_match_index}")
        app.logger.debug(f"Best match distance: {face_distances[best_match_index]}")

        if face_distances[best_match_index] < (1 - threshold):
            app.logger.debug(f"Face recognized: {known_face_names[best_match_index]}")
            return [(known_face_names[best_match_index], 1 - face_distances[best_match_index])]

    app.logger.debug("Face not recognized")
    return [("Unknown", None)]

def load_face_encodings_from_db():
    employees = Employee.query.all()
    known_face_encodings = []
    known_face_names = []
    for employee in employees:
        if employee.face_encoding:
            encoding = np.frombuffer(employee.face_encoding, dtype=np.float64)
            known_face_encodings.append(encoding)
            known_face_names.append(employee.name)
            app.logger.debug(f"Loaded encoding for {employee.name}: {encoding}")
    return known_face_encodings, known_face_names

def generate_frames(camera_url):
    app.logger.info(f"Starting to generate frames for {camera_url}")

    with app.app_context():
        known_face_encodings, known_face_names = load_face_encodings_from_db()

    bytes_data = bytes()
    frame_count = 0  # Counter for skipping frames

    while True:
        try:
            response = requests.get(camera_url, stream=True, timeout=10)
            if response.status_code != 200:
                app.logger.error(f"Failed to access camera at {camera_url}: {response.status_code}")
                time.sleep(5)
                continue

            for chunk in response.iter_content(chunk_size=1024):
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b + 2]
                    bytes_data = bytes_data[b + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    if frame is None:
                        app.logger.warning(f"Failed to decode frame from {camera_url}")
                        continue

                    # Skip frame processing for smoother streaming
                    frame_count += 1
                    if frame_count % 5 == 0:  # Process every 5th frame
                        frame = cv2.resize(frame, (640,480))
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_encodings, face_locations = get_face_embedding(rgb_frame)

                        app.logger.info(f"Detected {len(face_locations)} faces in the frame")

                        for face_encoding, face_location in zip(face_encodings, face_locations):
                            x1, y1, x2, y2 = face_location
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            recognized = recognize_face(face_encoding, known_face_encodings, known_face_names, threshold=0.7)
                            for name, similarity in recognized:
                                label = f"{name}: {similarity:.2f}" if similarity else "Unknown"
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                                if name != "Unknown":
                                    with app.app_context():
                                        record_attendance(name)

                    # Encode frame and send it
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except requests.exceptions.RequestException as e:
            app.logger.error(f"Request exception during streaming: {e}")
            time.sleep(5)

        except cv2.error as e:
            app.logger.error(f"OpenCV error during frame processing: {e}")

        except Exception as e:
            app.logger.error(f"Unexpected error during streaming: {e}")

        app.logger.info("Reconnecting...")
        bytes_data = bytes()

@app.route('/')
@app.route('/index')
def index():
    num_cameras = len(session.get('camera_urls', []))
    camera_urls = session.get('camera_urls', [])
    return render_template('index.html', num_cameras=num_cameras, camera_urls=camera_urls)

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    camera_urls = session.get('camera_urls', [])
    if 0 <= camera_id < len(camera_urls):
        return Response(generate_frames(camera_urls[camera_id]), mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Camera not found", 404


@app.route('/update_cameras', methods=['POST'])
def update_cameras():
    num_cameras = int(request.form.get('num_cameras', 0))
    camera_urls = [request.form.get(f'camera_url_{i}', '') for i in range(num_cameras)]
    session['camera_urls'] = camera_urls
    return redirect(url_for('index'))

@app.route('/add_employee', methods=['GET', 'POST'])
def add_employee():
    if request.method == 'POST':
        name = request.form['name']
        position = request.form['position']
        phone = request.form['phone']
        email = request.form['email']
        job_type = request.form['job_type']
        checkin_time = request.form['checkin_time']
        checkout_time = request.form['checkout_time']
        photo = request.files['photo']

        save_employee_data(name, position, phone, email, job_type, photo, checkin_time, checkout_time)
        flash('Employee added successfully!', 'success')
        return redirect(url_for('index'))

    return render_template('add_employee.html')

@app.route('/report')
def report():
    attendance_records = Attendance.query.all()
    return render_template('report.html', attendance_records=attendance_records)

if __name__ == '__main__':
    app.run(debug=True)
