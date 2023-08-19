import math

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
wgan_generator_path = "saved_model/wgan_generator"
wgan_discriminator_path = "saved_model/wgan_discriminator"


caae_discriminator_weight = 0.1
wgan_discriminator_weight = 0.05

batch_size = 64

dropout_ratio = 0.25

weight_decay = None

sample_image = "sample_image.jpg"
sample_decoded_image = "sample_decoded_image.jpg"

label_smoothing_ratio = 0.1
label_smoothing_logit_threshold = math.log(label_smoothing_ratio/(1-label_smoothing_ratio))

learning_rate = 0.0001
gradient_clip_norm = 1.0

kernal_clip_value = 0.05