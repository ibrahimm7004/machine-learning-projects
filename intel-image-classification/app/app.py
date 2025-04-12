from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the pre-trained model
model = load_model(
    'intel_model.h5')

# Function to preprocess the uploaded image


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file from the user
        file = request.files['file']

        # Save the file temporarily
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        # Map predicted class to category name
        categories = ["buildings", "forest",
                      "glacier", "mountain", "sea", "street"]
        predicted_category = categories[predicted_class]

        # Remove the temporary uploaded file
        os.remove(file_path)

        return render_template('result.html', prediction=predicted_category)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
