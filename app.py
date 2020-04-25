from flask import Flask,render_template,url_for,request
import pickle
import cv2
import tensorflow as tf
import os
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

#file upload preparation
CATEGORIES = ["Campus-Alpha", "Campus-Beta", "Campus-Gama"]
def prepare(filepath):
    print("filepath: " + filepath)
    IMG_SIZE_X = 300
    IMG_SIZE_Y = 40
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE_X, IMG_SIZE_Y))
    return new_array.reshape(-1, IMG_SIZE_X, IMG_SIZE_Y, 1)


#model._make_predict_function()
app = Flask(__name__)

session = tf.Session(graph=tf.Graph())
with session.graph.as_default():
    set_session(session)
    model = load_model("64x3-CNN-2.model")



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        file = request.files["file"]
        #file.save(os.path.join("uploads", file.filename))
        file.save(file.filename)
        print("file.filename: " + file.filename)
        #global sess
        #global graph
        with session.graph.as_default():
            set_session(session)
            prediction = model.predict([prepare(file.filename)])
            my_prediction = CATEGORIES[int(prediction[0][0])]
    return render_template('result.html',message = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)