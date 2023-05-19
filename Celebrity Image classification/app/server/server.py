from flask import Flask, request, jsonify
import util
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/hello')
def hello():
    return "hi"


@app.route('/classify_image', methods=['GET', 'POST'])
def classify_image():
    image_data = request.form['image_data']
    response = jsonify(util.classify_image(image_data))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    print("Starting python flask server for face detection ..... ")
    util.load_artifacts()
    app.run(port=5500)
