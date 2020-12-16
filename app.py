from flask import Flask, request
from image import ImageProvider, prepare_image_base64

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Welcome to Sandstorm'


@app.route('/mustaches', methods=['GET'])
def put_mustaches():
    image_url = request.args.get('url')
    print(image_url)

    with ImageProvider(image_url, 'hello') as image_path:
        return prepare_image_base64('images/zedge.png')


if __name__ == '__main__':
    app.run(threaded=True, port=5000)