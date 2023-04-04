from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2

UPLOAD_FOLDER = "./uploads/"
ALLOWED_EXTENSIONS = set(["jpg", "png"])
modelo = load_model("modeloBrain.h5")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXTENSIONS


def predict(filepath):
    IMG_SIZE = 200  # 50 in txt-based
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    arr = new_array.reshape (-1, IMG_SIZE, IMG_SIZE, 1)
    arreglo = modelo.predict(arr)
    resultado = arreglo.argmax(axis=-1)

    return "Tumor" if resultado[1]==1 else "Sano"


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route('/')
def index():
    return "Hello world"


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        if "ourfile" not in request.files:
            return "The form has no file part"
        f = request.files["ourfile"]
        if f.filename == "":
            return "No file selected"
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return redirect(url_for("get_file", filename=filename))
    return """
    <form method=POST enctype="multipart/form-data">
    <input type="file" name="ourfile">
    <input type="submit" value="UPLOAD">
    </form>
    """


@app.route("/uploads/<filename>")
def get_file(filename):
    print(send_from_directory(app.config["UPLOAD_FOLDER"], filename))
    prediccion = predict(f"uploads/{filename}")
    return prediccion


if __name__ == '__main__':
    app.run(debug=True,
            port=5005)
