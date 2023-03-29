from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import traceback
import json

application = Flask(__name__)

# Загрузка предварительно обученной модели VGG16
model = keras.applications.vgg16.VGG16(weights='imagenet')


def format_results(results):
    result_str = ""
    for i, res in enumerate(results):
        result_str += f"{i+1}. {res['_']} ({res['probability']*100:.2f}%)\n"
    return result_str

# Функция для классификации изображения
def classify_image(file_storage):
    # Преобразование изображения в массив numpy и нормализация
    img = Image.open(io.BytesIO(file_storage.read()))
    img = img.resize((224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Классификация изображения
    preds = model.predict(x)
    results = decode_predictions(preds, top=3)[0]

    # Возврат топ-3 результатов
    return [{'label': label,'_': _, 'probability': float(prob)} for label, _, prob in results]


@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predict', methods=['POST'])
def predict():
    try:
        # Получение изображения из запроса
        image = request.files['file']

        # Классификация изображения
        results = classify_image(image)

        # Форматирование результата
        formatted_results = format_results(results)

        # Возврат результата в шаблон
        return render_template('result.html', result=formatted_results)
    except Exception as e:
        traceback.print_exc()
        return f"An error occurred during prediction: {str(e)}"


if __name__ == '__main__':
    application.run(host='0.0.0.0')