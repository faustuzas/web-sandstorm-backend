import cv2
import numpy as np

eye_cascade = cv2.CascadeClassifier('eye.xml')
face_cascade = cv2.CascadeClassifier('face.xml')

glass_img = cv2.imread('g3.jpg')


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def wear_a_glass(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eye_dict = {}
    centers = []
    areas = []
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # iterating over the face detected
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_color)

        for j, (ex, ey, ew, eh) in enumerate(eyes):
            center = (x + int(ex + 0.5 * ew), y + int(ey + 0.5 * eh))
            area = ew * eh
            eye_dict[j] = {"center": center, "area": area}
            centers.append((x + int(ex + 0.5 * ew), y + int(ey + 0.5 * eh)))
            areas.append(ew * eh)
        detected_eyes = {}
        idx = 0
        for k, value in eye_dict.items():
            for j in range(1, len(eye_dict.keys())):
                if k + j < len(eye_dict.keys()):
                    detected_eyes[idx] = {
                        'pairs': (k, k + j),
                        'area_diff': abs(
                            (eye_dict[k]['area'] - eye_dict[k + j]['area']) / min(eye_dict[k]['area'],
                                                                                  eye_dict[k + j]['area'])
                        ),
                        'dist_diff': abs(eye_dict[k]['center'][1] - eye_dict[k + j]['center'][1]),
                        "avg_area": (eye_dict[k]['area'] + eye_dict[k + j]['area']) / 2,
                        "avg_dist": (eye_dict[k]['center'][1] + eye_dict[k + j]['center'][1]) / 2
                    }
                    idx += 1

        detected_eyes_sorted_by_area_diff = {
            k: v for k, v in sorted(
                detected_eyes.items(), key=lambda item: item[1]['area_diff']
            )
        }
        detected_eyes_sorted_by_avg_dist = {
            k: v for k, v in sorted(
                detected_eyes.items(), key=lambda item: item[1]['avg_dist']
            )
        }

        ranked_keys = {}
        for key in detected_eyes.keys():
            ranked_keys[key] = list(detected_eyes_sorted_by_area_diff.keys()).index(key) + list(
                detected_eyes_sorted_by_avg_dist.keys()).index(key)
        ranked_keys = sorted(ranked_keys, key=ranked_keys.get)
        best_keys_as_eyes = detected_eyes[ranked_keys[0]]['pairs'] if ranked_keys else ()
        centers = [eye_dict[i]['center'] for i in best_keys_as_eyes]

        if len(centers) > 0:
            # change the given value of 2.15 according to the size of the detected face
            glasses_width = 2.16 * abs(centers[1][0] - centers[0][0])
            overlay_img = np.ones(image.shape, np.uint8) * 255
            h, w = glass_img.shape[:2]
            scaling_factor = glasses_width / w

            overlay_glasses = cv2.resize(glass_img, None, fx=scaling_factor, fy=scaling_factor,
                                         interpolation=cv2.INTER_AREA)

            x = centers[0][0] if centers[0][0] < centers[1][0] else centers[1][0]
            y = centers[0][1] if centers[0][1] < centers[1][1] else centers[1][1]
            # The x and y variables below depend upon the size of the detected face.
            x -= 0.26 * overlay_glasses.shape[1]
            y -= 0.40 * overlay_glasses.shape[0]

            # Slice the height, width of the overlay image.
            h, w = overlay_glasses.shape[:2]
            overlay_img[int(y):int(y + h), int(x):int(x + w)] = overlay_glasses
            delta_y = centers[1][1] - centers[0][1]
            delta_x = centers[0][0] - centers[1][0]
            angle = np.arctan(delta_y / delta_x) * 90 / np.pi
            test_img = rotate_image(overlay_img, angle)
            overlay_img = test_img
            # Create a mask and generate it's inverse.
            gray_glasses = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray_glasses, 110, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            temp = cv2.bitwise_and(image, image, mask=mask)

            temp2 = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)
            final_img = cv2.add(temp, temp2)
            # imS = cv2.resize(final_img, (1366, 768))
            # cv2.imshow('Lets wear Glasses', final_img)
            cv2.imwrite(image_path, final_img)

            # cv2.waitKey()
            cv2.destroyAllWindows()
            break
