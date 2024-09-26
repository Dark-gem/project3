from try2 import app, db

# Define the Attendance model with extend_existing=True
class Attendance(db.Model):
    __tablename__ = 'attendance'
    id = db.Column(db.Integer, primary_key=True)
    employee_name = db.Column(db.String(100))
    year = db.Column(db.Integer)
    month = db.Column(db.Integer)
    day = db.Column(db.Integer)
    checkin_time = db.Column(db.String(20))
    checkout_time = db.Column(db.String(20))
    __table_args__ = {'extend_existing': True}

# Create tables within the Flask application context
def create_database():
    with app.app_context():
        db.create_all()
        print("Database created successfully!")

if __name__ == "__main__":
    create_database()
