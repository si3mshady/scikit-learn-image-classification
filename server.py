from flask import Flask, request, jsonify
import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('image_classifier.pkl', 'rb') as file:
    model_data = pickle.load(file)
    classifier = model_data['classifier']

# Define the categories
categories = ['empty', 'not_empty']

@app.route('/classify', methods=['POST'])
def classify_image():
    # Receive image file from the request
    image_file = request.files['image']

    # Read and preprocess the image
    img = imread(image_file)
    img_resize = resize(img, (15, 15))
    img_flat = img_resize.flatten()
    image_data = np.asarray([img_flat])

    # Make predictions
    prediction = classifier.predict(image_data)[0]
    category = categories[prediction]

    # Return the classification result as JSON
    return jsonify({'category': category})

if __name__ == '__main__':
    app.run(debug=True)


# curl -X POST -F "image=@words.jpeg" http://localhost:5000/classify

# venv) (base) Elliotts-Air:image_classifier_sklearn elliottarnold$ curl -X POST -F "image=@words.jpeg" http://localhost:5000/classify
# {
#   "category": "not_empty"
# (venv) (base) Elliotts-Air:image_classifier_sklearn elliottarnold$ curl -X POST -F "image=@empty_spot.jpg" http://localhost:5000/classify
# {
#   "category": "empty"
# }
# (venv) (base) Elliotts-Air:image_classifier_sklearn elliottarnold$ curl -X POST -F "image=@black_image.jpeg" http://localhost:5000/classify
# {
#   "category": "not_empty"
# }
# (venv) (base) Elliotts-Air:image_classifier_sklearn elliottarnold$ curl -X POST -F "image=@words.jpeg" http://localhost:5000/classify
# {
#   "category": "not_empty"

Fit Predict, 
these 2 methods preform the 