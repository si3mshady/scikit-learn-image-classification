# Image Classification Flask Application

This Flask application utilizes the power of Scikit-learn for image classification tasks. It allows you to train a model using Support Vector Machines (SVM) and serve it as an API for making image classification predictions.

## Installation

1. Clone this repository: `git clone <repository_url>`
2. Navigate to the project directory: `cd image-classification-flask-app`
3. Install the required dependencies: `pip install -r requirements.txt`

## Usage

1. Train the image classification model:
   - Prepare your image dataset and organize it into separate directories based on different categories.
   - Update the `PATH` variable in the `app.py` file to point to the directory containing your image dataset.
   - Run the training script: `python train.py`

2. Start the Flask server:
   - Run the following command: `python app.py`
   - The server will start running on `http://localhost:5000`

3. Make a prediction using cURL:
   - Open a new terminal window and run the following command:
     ```shell
     curl -X POST -F "image=@path/to/your/image.jpg" http://localhost:5000/classify
     ```
     Replace `path/to/your/image.jpg` with the actual path to the image file you want to classify.

   - You will receive a JSON response containing the predicted category for the provided image.

## Example Response

```json
{   "category": "not_empty" }
or 
{   "category": "empty" }