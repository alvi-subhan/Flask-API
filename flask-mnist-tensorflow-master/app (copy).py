from flask import Flask, request, render_template, jsonify

import tensorflow as tf
import keras
import os
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

from io import BytesIO
import base64
import numpy as np
from PIL import Image

from tf import softmax_predict, sigmoid_5_layers_predict, relu_5_layers_predict, conv2d_predict

app = Flask('flask-mnist-tensorflow')

app.config.from_pyfile('settings.py')


def create_model():
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(10, activation='softmax'))
  
  model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  return model


model=creat_model()
path='model.ckpt'
model.load_weights(path)

def decode_img():
    img = request.form['img']
    img = img.split("base64,")[1]
    img = BytesIO(base64.b64decode(img))
    img = Image.open(img)  # 读取图像
    img = Image.composite(img, Image.new('RGB', img.size, 'white'), img)
    img = img.convert('L')  # 转为灰度
    img = img.resize((28, 28), Image.ANTIALIAS)  # 压缩为28*28
    img = 1 - np.array(img, dtype=np.float32) / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    img = decode_img()
    y=model.predict(img)
    print(y)
    """return jsonify({
        'conv2d': conv2d_predict(img)[0].tolist(),
        'relu_5_layers': relu_5_layers_predict(img)[0].tolist(),
        'sigmoid_5_layers': sigmoid_5_layers_predict(img)[0].tolist(),
        'softmax': softmax_predict(img)[0].tolist(),
    })
"""

if __name__ == '__main__':
    app.run('0.0.0.0', 4000, debug=True)