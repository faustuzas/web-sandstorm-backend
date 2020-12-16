from random import random

from flask import Flask, request, send_file
from image import ImageProvider
from face_detection import add_mustaches

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Welcome to Sandstorm'


@app.route('/mustaches', methods=['GET'])
def put_mustaches():
    image_url = request.args.get('url')
    with ImageProvider(image_url, random() * 1000) as image_path:
        try:
            add_mustaches(image_path)
        except Exception:
            print('could not :/')

        return send_file(image_path)


if __name__ == '__main__':
    app.run(threaded=True, port=5000)