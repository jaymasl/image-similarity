# app.py
from flask import Flask, render_template, request
from skimage.metrics import structural_similarity as ssim

import numpy as np
import cv2

app = Flask(__name__)
application = app

@app.route('/')
def upload_form():
    # Render the upload form
    return render_template('upload.html', similarity=None)

@app.route('/compare', methods=['POST'])
def compare_images():
    # Get the uploaded files
    image1_file = request.files['image1']
    image2_file = request.files['image2']

    # Read the images using OpenCV
    image1 = cv2.imdecode(np.frombuffer(image1_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(np.frombuffer(image2_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Resize images to the same size for comparison
    gray1 = cv2.resize(gray1, (300, 300))
    gray2 = cv2.resize(gray2, (300, 300))

    # Calculate the Structural Similarity Index (SSI)
    similarity_index, _ = ssim(gray1, gray2, full=True)

    # Convert similarity index to percentage
    similarity_percentage = f"{(similarity_index * 100):.2f}"

    # Render the upload form with the similarity result
    return render_template('upload.html', similarity=similarity_percentage)

if __name__ == '__main__':
    app.run(debug=True)
