train_dataset_path = "dataset/train"
validation_dataset_path = "dataset/validation"
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

processed_train_dataset_path = "processed_dataset/train"
processed_validation_dataset_path = "processed_dataset/validation"

feature_size = 512

encoder_path = "saved_model/encoder"
decoder_path = "saved_model/decoder"
encoder_discriminator_path = "saved_model/encoder_discriminator"
decoder_discriminator_path = "saved_model/decoder_discriminator"
classifier_path = "saved_model/classifier"

discriminator_weight = 1

batch_size = 64

dropout_ratio = 0

weight_decay = None

sample_image = "sample_image.jpg"
sample_decoded_image = "sample_decoded_image.jpg"

soft_label_ratio = 0.1

learning_rate = 0.0001
clipnorm = 1.0