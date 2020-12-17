import cv2

from overlay_image import overlay_transparent

mustaches_img = cv2.imread('images/mustaches.png', -1)


def resize_mustache(image, width, height):
    return cv2.resize(image, (width, height))


mouth_cascade = cv2.CascadeClassifier('cascades/mouth_cascade.xml')


def add_mustaches(image_path):
    image = cv2.imread(image_path, -1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
    for (x, y, w, h) in mouth_rects:
        y = int(y - 0.15 * h)
        mustaches = resize_mustache(mustaches_img, w * 2, h * 2)
        image = overlay_transparent(image, mustaches,  x - int(w / 2), y - int(0.8 * h))

    cv2.imwrite(image_path, image)


def add_glasses(image_path):
    pass
