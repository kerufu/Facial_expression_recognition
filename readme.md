Expolore application of GAN in classification, experiment in CAAE and WGAN

Reason:
1. GAN based data augmentation, more precisely train encoder and classifier (the only two module actually running after deployment), achieve better performance with the same model scale
2. with finer training, kernal regularization and dropout (redundant neurals) can be removed, the scale of model to train increases but scale of  model to inference decreases

Test procedure:
1. Disable discriminator and scale dowm model until overfitting and test acc significantly degenerates
2. Gradually add weight on discriminator to mitigate overfitting

the encoded features are in range (-1, 1), the conditional label (expression class) are in one hot coding, e.g. 1, -1 , -1, ...

dataset: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset?resource=download

put the dataset in ./dataset

data are normalized into range (-1, 1), train on the whole train dataset and use validation dataset as test dataset

Docker usage:

To build image: docker build . -t tensorflow_opencv

To create container and start: docker run --name fer -e PYTHONUNBUFFERED=1 -it --gpus all tensorflow_opencv

To copy files from/to container: check the .sh scripts

To set up GPU on windows docker:
1. upgrade nvidia driver
2. wsl --install in administrator mode
3. enable wsl2 and ubuntu in docker
4. follow the instruction: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#install-guide

More reference:

https://medium.com/@KNuggies/tensorflow-with-gpu-on-windows-with-wsl-and-docker-75fb2edd571f

https://github.com/soumith/ganhacks

https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/

https://davidleonfdez.github.io/gan/2022/05/17/gan-convergence-stability.html

Hints:
1. Shuffle and batch tf.data.Dataset carefully
2. "selu" ≈ "BatchNormalization" + "leaky_relu", don't use "selu" + "BatchNormalization"
3. Gradient penalty or label smoothing when discriminator is unstable
4. Wasserstein loss is compatible with cross entropy from logit
5. Maxpooling for classification cnn, stride for generation cnn