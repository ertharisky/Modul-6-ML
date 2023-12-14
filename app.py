from flask import Flask, request, render_template, jsonify
from keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image as tf_image
from datetime import datetime

app = Flask(__name__)

model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Ambil waktu awal prediksi
        start_time = datetime.now()

        # Ambil file gambar dari permintaan POST
        file = request.files['file']

        # Lakukan preprocessing pada gambar
        img = Image.open(file).convert('RGB').resize((150, 150))
        img_array = tf_image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Lakukan prediksi dengan model
        prediction = model.predict(img_array)

        # Ambil waktu akhir prediksi
        end_time = datetime.now()

        # Hitung lama waktu prediksi
        prediction_time = end_time - start_time

        # Ambil label yang diprediksi
        predicted_label = str(np.argmax(prediction))
        labels = ['paper', 'rock', 'scissors']
        actual_label = labels[int(predicted_label)]

        # Ambil nama file gambar yang diprediksi
        image_name = file.filename

        # Hitung akurasi prediksi
        accuracy = prediction[0][int(predicted_label)] * 100.0

        # Return hasil prediksi dan informasi lainnya dalam bentuk JSON
        return jsonify({
            'prediction': prediction.tolist(),
            'predicted_label': predicted_label,
            'actual_label': actual_label,
            'accuracy': accuracy,
            'prediction_time': str(prediction_time),
            'image_name': image_name
        })


if __name__ == '__main__':
    app.run(debug=True)
