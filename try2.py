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
from retinaface import RetinaFace

# Initialize Flask application
app = Flask(__name__)

# Initialize face recognition model (Facenet)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/DELL/PycharmProjects/project3/instance/employees.db'
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

def detect_faces_retina(frame):
    """Detect faces using RetinaFace"""
    faces = RetinaFace.detect_faces(frame)
    boxes = []
    if faces is not None:
        for key in faces.keys():
            identity = faces[key]
            facial_area = identity['facial_area']
            boxes.append(facial_area)  # Format: [x1, y1, x2, y2]
    return boxes

def recognize_faces(frame, boxes):
    recognized_faces = []
    if boxes is not None:
        faces = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))
            face = face.astype(np.float32) / 255.0
            face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0)
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

def load_face_encodings_from_db():
    employees = Employee.query.all()
    known_face_encodings = []
    known_face_names = []
    for employee in employees:
        if employee.face_encoding:
            encoding = np.frombuffer(employee.face_encoding, dtype=np.float64)
            known_face_encodings.append(encoding)
            known_face_names.append(employee.name)
    return known_face_encodings, known_face_names

def generate_frames(camera_url):
    def frame_processing_thread(camera_url, known_face_encodings, known_face_names):
        bytes_data = bytes()
        frame_count = 0  # Counter for skipping frames

        while True:
            try:
                response = requests.get(camera_url, stream=True, timeout=10)
                if response.status_code != 200:
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
                            continue

                        # Skip frame processing for smoother streaming
                        frame_count += 1
                        if frame_count % 5 == 0:
                            frame = cv2.resize(frame, (640, 480))
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            face_boxes = detect_faces_retina(rgb_frame)

                            for box in face_boxes:
                                x1, y1, x2, y2 = box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                # Recognition part
                                recognized = recognize_faces(rgb_frame, face_boxes)
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
                time.sleep(5)

    known_face_encodings, known_face_names = load_face_encodings_from_db()
    thread = threading.Thread(target=frame_processing_thread, args=(camera_url, known_face_encodings, known_face_names))
    thread.start()


def save_employee_data(name, position, phone, email, job_type, photo, checkin_time, checkout_time):
    photo_filename = secure_filename(photo.filename) if photo else None
    face_encoding = None
    if photo:
        # Save the photo to the designated upload folder
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], photo_filename)
        photo.save(photo_path)

        # Generate face encoding from the uploaded photo
        image = face_recognition.load_image_file(photo_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            face_encoding = encodings[0].tobytes()  # Convert the encoding to a byte array for storage

    # Create a new Employee record
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

    # Save the employee record to the database
    db.session.add(new_employee)
    db.session.commit()


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

@app.route('/submit_employee', methods=['POST'])
def submit_employee():
    name = request.form['name']
    position = request.form['position']
    phone = request.form['phone']
    email = request.form['email']
    job_type = request.form['job_type']
    checkin_time = request.form.get('checkin_time', None)
    checkout_time = request.form.get('checkout_time', None)
    photo = request.files['photo']

    save_employee_data(name, position, phone, email, job_type, photo, checkin_time, checkout_time)
    flash('Employee added successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/report')
def report():
    attendance_records = Attendance.query.all()
    return render_template('report.html', attendance_records=attendance_records)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    app.run(debug=True)
