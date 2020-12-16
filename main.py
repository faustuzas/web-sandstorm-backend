from flask import Flask, request
from image import ImageProvider, prepare_image_base64

app = Flask(__name__)


@app.route('/', methods=['POST'])
def put_mustaches():
    data = request.get_json(force=True)

    with ImageProvider(data['imageUrl'], 'hello') as image_path:
        return prepare_image_base64(image_path)


if __name__ == '__main__':
    app.run(threaded=True, port=5000)
