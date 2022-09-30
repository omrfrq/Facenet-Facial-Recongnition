from architecture import *
from tensorflow.keras.models import load_model
from scipy.spatial.distance import cosine
from sklearn.preprocessing import Normalizer
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import pickle
import cv2 
import numpy as np
import mtcnn

######################################################################################################

data_set = 'dataset/'
shape = (160,160)
encoder = InceptionResNetV2()
encoder.load_weights("facenet_keras_weights.h5")
face_detector = mtcnn.MTCNN()
l2_normalizer = Normalizer('l2')
confidence=0.99
recognition=0.5
encodings = []
encoding_dict = dict()

###normalizer###
def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def get_encoding(encoder, face, size):

    face = normalize(face)
    face = cv2.resize(face, size)
    encoding = encoder.predict(np.expand_dims(face, axis=0))[0]
    return encoding

###bounding box###
def get_face(camera_frames, box):

    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = camera_frames[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

####loading encoding through pickle###
def load_encoding(path):

    with open(path, 'rb') as f:
        encoding= pickle.load(f)
    return encoding

###function for face detection ###
def detect_face(camera_frames ,face_detector , encoder, encoding):

    RGB_image = cv2.cvtColor(camera_frames, cv2.COLOR_BGR2RGB)
    result = face_detector .detect_faces(RGB_image)

    for r in result:

        if r['confidence'] < confidence:
            continue

        face, pt_1, pt_2 = get_face(RGB_image, r['box'])
        encode = get_encoding(encoder, face, shape)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding.items():
            dist = cosine(db_encode, encode)
            if dist < recognition and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':

            cv2.rectangle(camera_frames, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(camera_frames, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        else:

            cv2.rectangle(camera_frames, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(camera_frames, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
                        
    return camera_frames 

###############################################################################################################

###training###
for person_names in os.listdir(data_set):
    
    ###label is extracted from directory name###
    person = os.path.join(data_set,person_names)

    for image_name in os.listdir(person):
        image_path = os.path.join(person,image_name)

        image = cv2.imread(image_path)
        RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x = face_detector.detect_faces(RGB_image)
        x1, y1, width, height = x[0]['box']
        x1, y1 = abs(x1) , abs(y1)
        x2, y2 = x1+width , y1+height
        face = RGB_image[y1:y2 , x1:x2]
        
        face = normalize(face)
        face = cv2.resize(face,shape)
        face_d = np.expand_dims(face, axis=0)
        encode = encoder.predict(face_d)[0]
        encodings.append(encode)

    if encodings:
        encode = np.sum(encodings, axis=0 )
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        encoding_dict[person_names] = encode
    
with open('encodings.pkl', 'wb') as file:
    pickle.dump(encoding_dict, file)

###facial recogniton###
encoding = load_encoding('encodings.pkl')

video_capture = cv2.VideoCapture(2)

while video_capture.isOpened():

        ret,camera_frames = video_capture.read()

        if ret != True:
            print("Error, camera could not be opened") 
            break
        
        ###calling facial recognition function###
        camera_frames= detect_face(camera_frames , face_detector , encoder , encoding)

        cv2.imshow('camera', camera_frames)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break