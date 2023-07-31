import glob
#database url: http://www.consortium.ri.cmu.edu/ckagree/

emotion_path = "Emotion/"

# grab the file paths of all images
file_paths = glob.glob(emotion_path + "/*/*/*.txt")

f_sad = open("sad.txt", "w")
f_happy = open("happy.txt", "w")
f_angry = open("angry.txt", "w")
f_other = open("other.txt", "w")

for i, file in enumerate(file_paths):
    print(i)
    emotion = open(file, "r")
    label = emotion.readline()
    label = int(float(label))
    if label == 1:
        f_angry.write(file + "\n")
    elif label == 5:
        f_happy.write(file + "\n")
    elif label == 6:
        f_sad.write(file + "\n")
    else:
        f_other.write(file + "\n")
    emotion.close()
f_angry.close()
f_happy.close()
f_sad.close()
f_other.close()