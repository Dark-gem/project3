import os
import sqlite3
from datetime import datetime
import requests
import time

# Database setup
conn = sqlite3.connect('users.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER NOT NULL,
        address TEXT NOT NULL,
        position TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

# Fixed number of images to capture
NUM_IMAGES = 3


def add_user(name, age, address, position):
    cursor.execute('''
        INSERT INTO users (name, age, address, position)
        VALUES (?, ?, ?, ?)
    ''', (name, age, address, position))
    conn.commit()
    return cursor.lastrowid


def capture_image(esp32_url, save_path):
    try:
        response = requests.get(esp32_url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"Failed to capture image. Status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Error capturing image: {e}")
        return False


def process_form_submission(form_data):
    name = form_data.get('name')
    age = form_data.get('age')
    address = form_data.get('address')
    position = form_data.get('position')

    user_id = add_user(name, age, address, position)

    base_dir = "captured_images"
    user_dir = os.path.join(base_dir, name)
    os.makedirs(user_dir, exist_ok=True)

    esp32_camera_url = "http://192.168.18.49/capture"

    captured_images = []
    for i in range(NUM_IMAGES):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{name}_{timestamp}_{i + 1}.jpg"
        image_path = os.path.join(user_dir, image_filename)

        if capture_image(esp32_camera_url, image_path):
            print(f"Image {i + 1} captured and saved to {image_path}")
            captured_images.append(image_path)
        else:
            print(f"Failed to capture image {i + 1} from ESP32")

        time.sleep(1)  # Wait 1 second between captures

    return user_id, captured_images


def handle_form_submission(form_data):
    try:
        user_id, captured_images = process_form_submission(form_data)
        return {
            "success": True,
            "message": f"User added with ID: {user_id}",
            "captured_images": captured_images,
            "num_images_captured": len(captured_images)
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error processing form: {str(e)}"
        }


# Example usage (this would be replaced by your actual web framework code)
if __name__ == "__main__":
    # Simulating form data
    sample_form_data = {
        "name": "John Doe",
        "age": "30",
        "address": "123 Main St, Anytown, USA",
        "position": "Engineer"
    }

    result = handle_form_submission(sample_form_data)
    print(result)

conn.close()