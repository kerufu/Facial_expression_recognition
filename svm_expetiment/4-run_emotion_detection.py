import os.path

import numpy as np
import cv2
import dlib
from sklearn import svm
from sklearn.externals import joblib
from calculate_features import cal_features

enum_reverse_emotions = {
    0: "other",
    1: "happy",
    2: "angry",
    3: "sad"
}

CASCADE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(cv2.__file__), "../cv2/data/"))

DETECT_PARAMS = {
    "scaleFactor": 1.1,
    "minNeighbors": 6,
    "flags": 0,
}

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)

	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords

def svm_read_problem(data_file_name):
  """
  svm_read_problem(data_file_name) -> [y, x]
 
  Read LIBSVM-format data from data_file_name and return labels y
  and data instances x.
  """
  prob_y = []
  prob_x = []
  for line in open(data_file_name):
    line = line.split(None, 1)
    # In case an instance with all zero features
    if len(line) == 1: line += ['']
    label, features = line
    xi = []
    for e in features.split():
      ind, val = e.split(":")
      xi.append(float(val))
    prob_y += [float(label)]
    prob_x += [xi]
  return (prob_y, prob_x)

def main():
  face_cascade = cv2.CascadeClassifier(os.path.join(
      CASCADE_DIR, "haarcascade_frontalface_alt2.xml"))
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

  model_happy = joblib.load("detect_happy.pkl")
  model_angry = joblib.load("detect_angry.pkl")
  model_sad   = joblib.load("detect_sad.pkl")

  while True:
    ret, image = cap.read()
    if not ret:
      break
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, **DETECT_PARAMS)

    for face in detected_faces:
      face = dlib.rectangle(face[0], face[1], face[0]+face[2], face[1]+face[3])
      shape = shape_to_np(predictor(gray, face))
      cv2.rectangle(image, (face.left(), face.top()), \
          (face.right(), face.bottom()), (0, 0, 255), 3)
      features = cal_features(shape)
      prediction_happy = int(model_happy.predict([features])[0])
      prediction_angry = int(model_angry.predict([features])[0])
      prediction_sad   = int(model_sad.predict([features])[0])
      if prediction_happy == 1:
        cv2.putText(image, enum_reverse_emotions[prediction_happy], \
            (face.left(), face.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, \
            (0, 255, 0), 2)
      elif prediction_angry == 2:
        cv2.putText(image, enum_reverse_emotions[prediction_angry], \
            (face.left(), face.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, \
            (0, 255, 0), 2)
      elif prediction_sad == 3:
        cv2.putText(image, enum_reverse_emotions[prediction_sad], \
            (face.left(), face.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, \
            (0, 255, 0), 2)

    cv2.imshow("emotion", image)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord("q"):
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
