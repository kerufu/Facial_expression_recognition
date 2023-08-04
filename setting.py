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

discriminator_weight = 0.3

batch_size = 128

dropout_ratio = 0.0

sample_image = "sample_image.jpg"
sample_decoded_image = "sample_decoded_image.jpg"