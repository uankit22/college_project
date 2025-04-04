from flask import Flask, request, send_file, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from contrast import Ying_2017_CAIP  # Importing the contrast enhancement function

# Flask app setup
app = Flask(__name__)

# File Upload Configuration
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Check file extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Home Route
@app.route("/")
def home():
    return jsonify({"message": "Welcome to Image Contrast Enhancer API!"})

# Upload & Process Image
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Read Image
    img = cv2.imread(filepath)
    enhanced_img = Ying_2017_CAIP(img)  # Enhance contrast

    # Save Processed Image
    processed_path = os.path.join(app.config["PROCESSED_FOLDER"], "enhanced_" + filename)
    cv2.imwrite(processed_path, enhanced_img)

    return send_file(processed_path, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)
