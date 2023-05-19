import cv2
import joblib
import json
import numpy as np
import base64
from wavelet import w2d

__class_name_to_number = {}
__class_number_to_name = {}
__model = None


def classify_image(image_base_64, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base_64)
    results = []
    for img in imgs:
        scaled_raw_image = cv2.resize(img, (32, 32))
        im_har = w2d(img, 'db1', 5)
        scaled_wav_image = cv2.resize(im_har, (32, 32))
        # raw image has 3 color channels
        combined_img = np.vstack((scaled_raw_image.reshape(32 * 32 * 3, 1), scaled_wav_image.reshape(32 * 32, 1)))
        len_img_array = 32 * 32 * 3 + 32 * 32
        final_img = combined_img.reshape(1, len_img_array).astype(float)

        results.append({
            'celeb' : __class_number_to_name[__model.predict(final_img)[0]],
            'class_probability' : np.round(__model.predict_proba(final_img)*100,2).tolist()[0]
        })

    return results

def load_artifacts():
    print("loading artifacts")
    global  __class_name_to_number
    global  __class_number_to_name

    with open("./artifacts/celeb_names_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}
        print('dictionary loading done ...')

    global __model

    with open("./artifacts/face_detection_model.pkl","rb") as f:
        __model = joblib.load(f)
        print("model loading done ...")


def get_cv2_image_from_base64_string(base64str):
    encoded_data = base64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img


def get_cropped_image_if_2_eyes(image_path=None, image_base64_data=None):
    face_cascade = cv2.CascadeClassifier('./artifacts/opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./artifacts/opencv/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    elif image_base64_data:
        img = get_cv2_image_from_base64_string(image_base64_data)
    else:
        return Exception("No image found")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces

def get_b64_from_img_messi():
    with open('./test_images/messi_base64.txt') as f:
        return f.read()


if __name__ == '__main__':
    load_artifacts()
    print(classify_image(get_b64_from_img_messi()))
    print(classify_image(None, './test_images/sharapova1.jpg'))

