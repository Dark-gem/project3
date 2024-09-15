import sys
import os

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from attendance_api import db, app  # Ensure your Flask app and db are imported correctly
from attendance_api import Employee  # Import the Employee model from the correct file

# Ensure the script runs in the context of the Flask application
with app.app_context():
    db.create_all()
    print("Database tables created.")
