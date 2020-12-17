import cv2
import numpy as np

from overlay_image import overlay_transparent

#
# face_cascade = cv2.CascadeClassifier('face_cascade.xml')
# smile_cascade = cv2.CascadeClassifier('smile_cascade.xml')
# eye_cascade = cv2.CascadeClassifier('eye_cascade.xml')
#
# # read both the images of the face and the glasses
mustaches_img = cv2.imread('images/mustaches.png', -1)


#
#
# # mustaches_img = cv2.imread('images/glasses.png')
#
#
# def add_mustaches(image_path):
#     image = cv2.imread(image_path, -1)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30)
#     )
#
#     result = image
#     for (x, y, w, h) in faces:
#         result = overlay_transparent(image, mustaches_img, x + int(w / 2), y + int(0.7 * h))
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     cv2.imwrite('result.jpg', result)
#
#
# add_mustaches('Donald-J-Trump-2010.jpg')

# def add_mustaches(image_path):
#     image = cv2.imread(image_path, -1)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     centers = []
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#
#     # iterating over the face detected
#     for (x, y, w, h) in faces:
#
#         # create two Regions of Interest.
#         roi_gray = gray[y:y + h, x:x + w]
#         roi_color = image[y:y + h, x:x + w]
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#
#         # Store the coordinates of eyes in the image to the 'center' array
#         for (ex, ey, ew, eh) in eyes:
#             centers.append((x + int(ex + 0.5 * ew), y + int(ey + 0.5 * eh)))
#
#     if len(centers) > 0:
#         # change the given value of 2.15 according to the size of the detected face
#         glasses_width = 2.16 * abs(centers[1][0] - centers[0][0])
#         overlay_img = np.ones(image.shape, np.uint8) * 255
#         h, w = mustaches_img.shape[:2]
#         scaling_factor = glasses_width / w
#
#         overlay_glasses = cv2.resize(mustaches_img, None, fx=scaling_factor, fy=scaling_factor,
#                                      interpolation=cv2.INTER_AREA)
#
#         x = centers[0][0] if centers[0][0] < centers[1][0] else centers[1][0]
#
#         # The x and y variables below depend upon the size of the detected face.
#         x -= 0.26 * overlay_glasses.shape[1]
#         y += 0.85 * overlay_glasses.shape[0]
#
#         # Slice the height, width of the overlay image.
#         h, w = overlay_glasses.shape[:2]
#         overlay_img[int(y):int(y + h), int(x):int(x + w)] = overlay_glasses
#
#         # Create a mask and generate it's inverse.
#         gray_glasses = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
#         ret, mask = cv2.threshold(gray_glasses, 110, 255, cv2.THRESH_BINARY)
#         mask_inv = cv2.bitwise_not(mask)
#         temp = cv2.bitwise_and(image, image, mask=mask)
#
#         temp2 = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)
#         final_img = cv2.add(temp, temp2)
#         cv2.imwrite(image_path, final_img)


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
