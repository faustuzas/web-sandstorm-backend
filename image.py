from urllib.request import urlretrieve
from urllib.parse import urlparse

import os
import base64


class ImageProvider:

    def __init__(self, url_, filename_to_save):
        self.saved_image_path = None

        self.url = url_
        self.filename_to_save = filename_to_save

    def __enter__(self):
        filename = self.filename_to_save + extract_url_extension(self.url)
        (saved_image_path, _) = urlretrieve(self.url, filename)

        self.saved_image_path = saved_image_path
        return self.saved_image_path

    def __exit__(self, type_, value, traceback):
        os.remove(self.saved_image_path)


def extract_url_extension(url_):
    path = urlparse(url_).path
    return os.path.splitext(path)[1]


def prepare_image_base64(image_path):
    with open(image_path, 'rb') as f:
        encoded_string = base64.b64encode(f.read())
        return encoded_string.decode('utf-8')
        # return f"data:image/png;base64, {encoded_string.decode('utf-8')}"