import sqlite3
from flask import Flask, render_template, Response, request, redirect, url_for, flash, session, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import requests
from datetime import datetime, timedelta, date
import pandas as pd
import face_recognition
from facenet_pytorch import InceptionResnetV1
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
import atexit
import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Define the function to handle late entries
def handle_late_entries():
    today = date.today()
    employees = Employee.query.all()
    for employee in employees:
        scheduled_checkin = datetime.strptime(employee.checkin_time, '%H:%M:%S').time()
        attendance = Attendance.query.filter_by(employee_id=employee.id, date=today).first()
        if attendance and attendance.checkin_time:
            checkin_time = datetime.strptime(attendance.checkin_time, '%H:%M:%S').time()
            if checkin_time > scheduled_checkin:
                attendance.late_entry = True
                db.session.commit()

# Initialize Flask application
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///C:/Users/DELL/PycharmProjects/project3/instance/employees.db')
app.config['SQLALCHEMY_BINDS'] = {
    'attendance': os.getenv('ATTENDANCE_DATABASE_URL', 'sqlite:///C:/Users/DELL/PycharmProjects/project3/instance/attendance.db')
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_unique_secret_key_here')

# Initialize SQLAlchemy and Migrate
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Initialize face recognition model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Create a scheduler instance
scheduler = BackgroundScheduler()
# Schedule the job to run daily at midnight using the cron trigger
scheduler.add_job(func=handle_late_entries, trigger=CronTrigger(hour=0, minute=0))
scheduler.start()

# Models
class Employee(db.Model):
    __tablename__ = 'employee'
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
    __bind_key__ = 'attendance'  # Bind to attendance.db
    __tablename__ = 'attendance'
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    status = db.Column(db.String(10), nullable=False)
    checkin_time = db.Column(db.String(10), nullable=True)
    checkout_time = db.Column(db.String(10), nullable=True)
    late_entry = db.Column(db.Boolean, default=False)
    early_exit = db.Column(db.Boolean, default=False)

# Define additional functions

def get_daily_present_employees():
    today = date.today()
    present_employees = Attendance.query.filter_by(date=today, status='Present').all()
    return [{'employee_id': a.employee_id, 'employee_name': Employee.query.get(a.employee_id).name} for a in present_employees]

def get_daily_absent_employees():
    today = date.today()
    absent_employees = db.session.query(Employee).outerjoin(Attendance, (Employee.id == Attendance.employee_id) & (Attendance.date == today))\
        .filter(Attendance.status.is_(None)).all()
    return [{'employee_id': e.id, 'employee_name': e.name} for e in absent_employees]

def preprocess_image(image):
    image = cv2.resize(image, (160, 160))
    image = image.astype(np.float32) / 255.0
    return torch.tensor(image).unsqueeze(0)

def predict_face(image):
    image = preprocess_image(image)
    embedding = facenet_model(image).detach().numpy()[0]
    return embedding

def compare_embeddings(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

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
        today = date.today()
        if employee.checkin_date != today:
            employee.checkin_time = datetime.now().time()
            employee.checkin_date = today
            db.session.commit()

def generate_report(start_date, end_date):
    attendances = Attendance.query.filter(Attendance.date.between(start_date, end_date)).all()
    present_employees = []
    absent_employees = []

    for attendance in attendances:
        employee = Employee.query.get(attendance.employee_id)
        record = {
            'id': employee.id,
            'date': attendance.date,
            'checkin_time': attendance.checkin_time,
            'checkout_time': attendance.checkout_time,
            'status': attendance.status,
            'early_exit': attendance.early_exit
        }
        if attendance.status == 'Present':
            present_employees.append(record)
        else:
            absent_employees.append(record)

    return present_employees, absent_employees

def record_attendance(employee_id, checkin_time=None, checkout_time=None):
    employee = Employee.query.get(employee_id)
    if not employee:
        return

    today = date.today()
    attendance = Attendance.query.filter_by(employee_id=employee_id, date=today).first()

    if attendance:
        if checkin_time:
            attendance.checkin_time = checkin_time
        if checkout_time:
            attendance.checkout_time = checkout_time
        attendance.status = 'Present' if checkin_time else 'Absent'
    else:
        status = 'Present' if checkin_time else 'Absent'
        attendance = Attendance(
            employee_id=employee_id,
            date=today,
            checkin_time=checkin_time,
            checkout_time=checkout_time,
            status=status
        )
        db.session.add(attendance)
    db.session.commit()

def generate_monthly_report(year, month):
    start_date = datetime(year, month, 1).date()
    end_date = (datetime(year, month + 1, 1).date() - timedelta(days=1)).date()

    attendances = Attendance.query.filter(Attendance.date.between(start_date, end_date)).all()
    data = [{
        'Employee Name': Employee.query.get(a.employee_id).name,
        'Date': a.date,
        'Check-in Time': a.checkin_time,
        'Check-out Time': a.checkout_time,
        'Status': a.status,
        'Late Entry': a.late_entry,
        'Early Exit': a.early_exit,
        'Job Type': Employee.query.get(a.employee_id).job_type
    } for a in attendances]

    df = pd.DataFrame(data)
    excel_filename = f'attendance_report_{year}_{month}.xlsx'
    df.to_excel(excel_filename, index=False)
    return excel_filename

def get_face_embedding(image):
    face_locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, face_locations)
    return encodings, face_locations

def recognize_face(face_encoding, known_face_encodings, known_face_names, threshold=0.6):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=threshold)
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        return known_face_names[best_match_index]
    return 'Unknown'

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/report')
def report():
    start_date = request.args.get('start_date', date.today())
    end_date = request.args.get('end_date', date.today())
    present_employees, absent_employees = generate_report(start_date, end_date)
    return render_template('report.html', present_employees=present_employees, absent_employees=absent_employees)

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
        flash('Employee added successfully!')
        return redirect(url_for('index'))
    return render_template('add_employee.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        # This should be replaced with your actual video stream code
        while True:
            success, frame = cap.read()  # Replace cap.read() with your video capture method
            if not success:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload/<filename>')
def upload(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/update_cameras', methods=['POST'])
def update_cameras():
    # Implement the logic for updating cameras here
    flash('Cameras updated successfully!')
    return redirect(url_for('index'))

@app.route('/update_attendance', methods=['POST'])
def update_attendance():
    employee_id = request.form['employee_id']
    checkin_time = request.form.get('checkin_time')
    checkout_time = request.form.get('checkout_time')
    record_attendance(employee_id, checkin_time, checkout_time)
    flash('Attendance updated successfully!')
    return redirect(url_for('index'))

# Shutdown scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())

if __name__ == '__main__':
    app.run(debug=True)
