train_dataset_path = "../dataset/train"
validation_dataset_path = "../dataset/validation"
expression_classes = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]
num_classes = len(expression_classes)

image_size = 48

processed_train_dataset_path = "proccessed_dataset/train"
processed_validation_dataset_path = "proccessed_dataset/validation"

feature_size = 16