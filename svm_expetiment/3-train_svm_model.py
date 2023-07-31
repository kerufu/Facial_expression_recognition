from sklearn import svm
from sklearn.externals import joblib

dataset = ["emotion_happy.train", "emotion_angry.train", "emotion_sad.train"]
names = {
    0: "detect_happy.pkl",
    1: "detect_angry.pkl",
    2: "detect_sad.pkl"
}

def svm_read_problem(data_file_name):
  prob_y = []
  prob_x = []
  for line in open(data_file_name):
    line = line.split(None, 1)
    # In case an instance with all zero features
    if len(line) == 1: line += [""]
    label, features = line
    xi = []
    for e in features[:-1].split():
      ind, val = e.split(":")
      xi.append(float(val))
    prob_y += [float(label)]
    prob_x += [xi]
  return (prob_y, prob_x)

for i, train in enumerate(dataset):
  Y, X = svm_read_problem(train)
  clf = svm.SVC(C=500.0)
  clf.fit(X, Y)
  joblib.dump(clf, names[i])
print("Successfully train the SVM model!")
