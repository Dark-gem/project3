

from flask import Flask, render_template, request, jsonify, Response
import os
import base64
import cv2
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)

# Configure the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///person.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    mobile_no = db.Column(db.String(20), nullable=False)
    address = db.Column(db.String(200), nullable=False)
    position = db.Column(db.String(100), nullable=False)
    images = db.relationship('Image', backref='user', lazy=True)


# Image model
class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_data = db.Column(db.Text, nullable=False)


# Initialize the database only once
db_initialized = False



@app.before_request
def create_tables():
    global db_initialized
    if not db_initialized:
        db.create_all()
        db_initialized = True


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/add_user', methods=['POST'])
def add_user():
    try:
        name = request.form['name']
        age = int(request.form['age'])
        mobile_no = request.form['mobile_no']
        address = request.form['address']
        position = request.form['position']

        # Create a new user instance
        new_user = User(name=name, age=age, mobile_no=mobile_no, address=address, position=position)
        db.session.add(new_user)
        db.session.commit()

        # Handle image uploads
        if 'images' in request.files:
            images = request.files.getlist('images')
            for image in images:
                if image:
                    image_data = base64.b64encode(image.read()).decode('utf-8')
                    new_image = Image(user_id=new_user.id, image_data=image_data)
                    db.session.add(new_image)
            db.session.commit()

        return jsonify({"message": "User added successfully"})
    except Exception as e:
        db.session.rollback()
        print(f"Error occurred: {e}")  # Print the error message to the console
        return jsonify({"error": str(e)}), 400


@app.route('/view_users', methods=['GET'])
def view_users():
    try:
        users = User.query.all()
        user_list = []
        for user in users:
            user_data = {
                'id': user.id,
                'name': user.name,
                'age': user.age,
                'mobile_no': user.mobile_no,
                'address': user.address,
                'position': user.position,
                'images': [image.image_data for image in user.images]
            }
            user_list.append(user_data)
        return jsonify(user_list)
    except Exception as e:
        print(f"Error occurred: {e}")  # Print the error message to the console
        return jsonify({"error": str(e)}), 400


@app.route('/delete_user/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        user = User.query.get(user_id)
        if user:
            # Delete associated images first
            Image.query.filter_by(user_id=user.id).delete()
            db.session.delete(user)
            db.session.commit()
            return jsonify({"message": "User deleted successfully"})
        else:
            return jsonify({"error": "User not found"}), 404
    except Exception as e:
        db.session.rollback()
        print(f"Error occurred: {e}")  # Print the error message to the console
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
