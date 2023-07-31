import numpy as np

def distance_between(p1, p2):
	return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def point_between(p1, p2):
	return np.array([abs(p1[0] - p2[0]) / 2.0 + min(p1[0], p2[0]), abs(p1[1] - p2[1]) / 2.0 + min(p1[1], p2[1])])

def cal_features(landmarks):
	# the shape of landmarks should be (68, 2)
	features = [0] * 89
	
	lmk = landmarks
	left_eye = point_between(lmk[36], lmk[39])
	right_eye = point_between(lmk[42], lmk[45])
	nose = point_between(lmk[30], lmk[33])
	between_eyes = distance_between(left_eye, right_eye)

	i = 0
	
	for x in range(0, 17):
		features[i] = distance_between(lmk[x], nose) / between_eyes
		i += 1
	for x in range(17, 22):
		features[i] = distance_between(lmk[x], left_eye) / between_eyes
		i += 1
	for x in range(22, 27):
		features[i] = distance_between(lmk[x], right_eye) / between_eyes
		i += 1
	for x in range(31, 36):
		features[i] = distance_between(lmk[x], nose) / between_eyes
		i += 1
	for x in range(36, 42):
		features[i] = distance_between(lmk[x], left_eye) / between_eyes
		i += 1
	for x in range(42, 48):
		features[i] = distance_between(lmk[x], right_eye) / between_eyes
		i += 1
	for x in range(48, 68):
		features[i] = distance_between(lmk[x], nose) / between_eyes
		i += 1
	for x in range(0, 5):
		features[i] = distance_between(lmk[17+x], lmk[26-x]) / between_eyes
		i += 1
	for x in range(17, 22):
		features[i] = distance_between(lmk[x], nose) / between_eyes
		i += 1
	for x in range(22, 27):
		features[i] = distance_between(lmk[x], nose) / between_eyes
		i += 1
	features[i] = distance_between(lmk[52], lmk[56]) / between_eyes
	i += 1
	features[i] = distance_between(lmk[51], lmk[57]) / between_eyes
	i += 1
	features[i] = distance_between(lmk[50], lmk[58]) / between_eyes
	i += 1
	features[i] = distance_between(lmk[48], lmk[54]) / between_eyes
	i += 1
	features[i] = distance_between(lmk[49], lmk[53]) / between_eyes
	i += 1
	features[i] = distance_between(lmk[59], lmk[55]) / between_eyes
	i += 1
	features[i] = distance_between(lmk[60], lmk[64]) / between_eyes
	i += 1
	features[i] = distance_between(lmk[63], lmk[65]) / between_eyes
	i += 1
	features[i] = distance_between(lmk[62], lmk[66]) / between_eyes
	i += 1
	features[i] = distance_between(lmk[61], lmk[67]) / between_eyes
	
	return features
