# Read path from "happy.txt", detect face in the image, find facial landmarks
# in the face and calculate features by normalizing distances between each 
# landmark. Do the same thing to "sad.txt", "angry.txt" and "other.txt". For
# "other.txt", I also add some neutral faces.

import os

import numpy as np
import cv2
import dlib

enum_emotions = {
    "other": 0,
    "happy": 1,
    "angry": 2,
    "sad": 3
}

CASCADE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(cv2.__file__), "../cv2/data/"))

DETECT_PARAMS = {
    "scaleFactor": 1.1,
    "minNeighbors": 6,
    "flags": 0,
}

class ImageProcessor:
  def __init__(self, landmark_model_path):
    self.face_cascade = self._load_face_detector()
    self.landmark_predictor = self._load_landmark_model(landmark_model_path)

  def _load_face_detector(self):
    return cv2.CascadeClassifier(os.path.join(
        CASCADE_DIR, "haarcascade_frontalface_alt2.xml"))

  def _load_landmark_model(self, landmark_model_path):
    return dlib.shape_predictor(landmark_model_path)

  def _shape_to_np(self, shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
      coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

  def _distance_between(self, p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

  def _point_between(self, p1, p2):
    return np.array([abs(p1[0] - p2[0]) / 2.0 + min(p1[0], p2[0]), \
        abs(p1[1] - p2[1]) / 2.0 + min(p1[1], p2[1])])

  def _calc_features(self, lmk, label, fout):
    # the shape of landmarks should be (68, 2)
    left_eye = self._point_between(lmk[36], lmk[39])
    right_eye = self._point_between(lmk[42], lmk[45])
    nose = self._point_between(lmk[30], lmk[33])
    between_eyes = self._distance_between(left_eye, right_eye)

    fout.write(str(label))
    i = 0
    for x in range(0, 17):
      fout.write(" " + str(i) + ":" + \
          str(self._distance_between(lmk[x], nose) / between_eyes))
      i += 1
    for x in range(17, 22):
      fout.write(" " + str(i) + ":" + \
          str(self._distance_between(lmk[x], left_eye) / between_eyes))
      i += 1
    for x in range(22, 27):
      fout.write(" " + str(i) + ":" + \
          str(self._distance_between(lmk[x], right_eye) / between_eyes))
      i += 1
    for x in range(31, 36):
      fout.write(" " + str(i) + ":" + \
          str(self._distance_between(lmk[x], nose) / between_eyes))
      i += 1
    for x in range(36, 42):
      fout.write(" " + str(i) + ":" + \
          str(self._distance_between(lmk[x], left_eye) / between_eyes))
      i += 1
    for x in range(42, 48):
      fout.write(" " + str(i) + ":" + \
          str(self._distance_between(lmk[x], right_eye) / between_eyes))
      i += 1
    for x in range(48, 68):
      fout.write(" " + str(i) + ":" + \
          str(self._distance_between(lmk[x], nose) / between_eyes))
      i += 1
    for x in range(0, 5):
      fout.write(" " + str(i) + ":" + \
          str(self._distance_between(lmk[17+x], lmk[26-x]) / between_eyes))
      i += 1
    for x in range(17, 22):
      fout.write(" " + str(i) + ":" + \
          str(self._distance_between(lmk[x], nose) / between_eyes))
      i += 1
    for x in range(22, 27):
      fout.write(" " + str(i) + ":" + \
          str(self._distance_between(lmk[x], nose) / between_eyes))
      i += 1
    fout.write(" " + str(i) + ":" + \
        str(self._distance_between(lmk[52], lmk[56]) / between_eyes))
    i += 1
    fout.write(" " + str(i) + ":" + \
        str(self._distance_between(lmk[51], lmk[57]) / between_eyes))
    i += 1
    fout.write(" " + str(i) + ":" + \
        str(self._distance_between(lmk[50], lmk[58]) / between_eyes))
    i += 1
    fout.write(" " + str(i) + ":" + \
        str(self._distance_between(lmk[48], lmk[54]) / between_eyes))
    i += 1
    fout.write(" " + str(i) + ":" + \
        str(self._distance_between(lmk[49], lmk[53]) / between_eyes))
    i += 1
    fout.write(" " + str(i) + ":" + \
        str(self._distance_between(lmk[59], lmk[55]) / between_eyes))
    i += 1
    fout.write(" " + str(i) + ":" + \
        str(self._distance_between(lmk[60], lmk[64]) / between_eyes))
    i += 1
    fout.write(" " + str(i) + ":" + \
        str(self._distance_between(lmk[63], lmk[65]) / between_eyes))
    i += 1
    fout.write(" " + str(i) + ":" + \
        str(self._distance_between(lmk[62], lmk[66]) / between_eyes))
    i += 1
    fout.write(" " + str(i) + ":" + \
        str(self._distance_between(lmk[61], lmk[67]) / between_eyes))
    fout.write("\n")

  def _detect_landmarks(self, gray, face):
    shape = self.landmark_predictor(gray, face)
    return self._shape_to_np(shape)

  def _detect_face(self, img_path):
    gray = cv2.imread(img_path, 0)
    detected_face = self.face_cascade.detectMultiScale(gray, **DETECT_PARAMS)[0]
    return gray, detected_face

  def generate_features(self, img_paths, label, file_name):
    fout = open(file_name, "a")
    for path in img_paths:
      gray, face = self._detect_face(path)
      face = dlib.rectangle(face[0], face[1], face[0]+face[2], face[1]+face[3])
      shape = self._detect_landmarks(gray, face)
      self._calc_features(shape, label, fout)
    fout.close()

  def generate_training_data(self, file_path, file_name):
    emotion_label = enum_emotions[file_path.split(".")[0]]
    img_paths = []
    with open(file_path, "r") as f:
      for path in f:
        pos = path.split("emotion")[0].split("Emotion")[1][:-1]
        img_paths.append("Images" + pos + ".png")
    self.generate_features(img_paths, emotion_label, file_name)
    img_paths = []
    with open(file_path, "r") as f:
      for path in f:
        neg = path.split("emotion")[0].split("Emotion")[1][:-3]
        img_paths.append("Images" + neg + "01.png")
    self.generate_features(img_paths, 0, file_name)
    '''
    if emotion_label == 0:
      with open(file_path, "r") as f:
        for path in f:
          pos = path.split("emotion")[0].split("Emotion")[1]
          img_paths.append("Images" + pos[:-1] + ".png")
          neutral = pos[:-3]
          img_paths.append("Images" + neutral + "01.png")
    else:
      with open(file_path, "r") as f:
        for path in f:
          pos = path.split("emotion")[0].split("Emotion")[1][:-1]
          img_paths.append("Images" + pos + ".png")
    self.generate_features(img_paths, emotion_label)
    '''

def main():
  landmark_path = "shape_predictor_68_face_landmarks.dat"
  image_processor = ImageProcessor(landmark_path)
  # Other emotions
  #image_processor.generate_training_data("other.txt")
  # Happy
  image_processor.generate_training_data("happy.txt", "emotion_happy.train")
  # Angry
  image_processor.generate_training_data("angry.txt", "emotion_angry.train")
  # Sad
  image_processor.generate_training_data("sad.txt", "emotion_sad.train")
  print("Success!")

if __name__ == "__main__":
  main()
