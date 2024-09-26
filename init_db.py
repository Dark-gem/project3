from your_flask_app import app, db  # Adjust import according to your app structure

with app.app_context():
    db.create_all()
