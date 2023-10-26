# Import necessary libraries
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image

# Load the fruit classification model
model_path = "models/fruits.h5"
model = load_model(model_path)

# Define a list of fruit classes
class_name = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']

# Flask setup
from flask import Flask, request, render_template
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# Define routes for the web application
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

# Function to predict spoilage duration based on fruit
def predict_spoilage_duration(fruit_class):
    # Define spoilage durations based on your provided information
    durations = {
        'freshbanana': '2-7 days at room temperature, 2-4 weeks in the fridge',
        'freshapples': '1-2 weeks at room temperature, several months in the fridge',
        'freshoranges': '1-2 weeks at room temperature, 1-2 months in the fridge'
    }
    
    # Get the duration based on the fruit class
    duration = durations.get(fruit_class, 'Duration information not available')
    
    return duration

# Route for handling fruit prediction
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        f = request.files['fruit']
        filename = f.filename
        target = os.path.join(APP_ROOT, 'images/')
        des = "/".join([target, filename])
        f.save(des)

        # Load and preprocess the test image
        test_image = image.load_img('images/' + filename, target_size=(300, 300))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Predict fruit class
        fruit_prediction = model.predict(test_image)
        predicted_class = class_name[np.argmax(fruit_prediction[0])]
        confidence = round(np.max(fruit_prediction[0]) * 100)

        # Predict spoilage duration based on the predicted fruit
        spoilage_duration = predict_spoilage_duration(predicted_class)

        return render_template("prediction.html", confidence="Chances: " + str(confidence) + "%", prediction="Prediction: " + str(predicted_class), spoilage_duration="Spoilage Duration: " + spoilage_duration)

    else:
        return render_template("prediction.html")

# Start the Flask application
if __name__ == '__main__':
    app.debug = True
    app.run()
