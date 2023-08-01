Conditional AE with discriminators on feature and reconstruction, as well as a classifier

Reason for using CAAE to encode image before classifying: more precisely train encoder and classifier (the only two module actually running after deployment), achieve better performance with the same model scale

Consider reduce the model scale and remove some regulations

the encoded features are in range (-1, 1), the conditional label (expression class) are in one hot coding, e.g. 1, -1 , -1, ...

dataset: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset?resource=download

put the dataset in ./dataset

data are normalized into range (-1, 1), train on the whole train dataset and use validation dataset as test dataset

Docker usage:

To build image: docker build . -t tensorflow_opencv

To create container and start: docker run -e PYTHONUNBUFFERED=1 --gpus all tensorflow_opencv bash

To copy files from/to container: https://support.sitecore.com/kb?id=kb_article_view&sysparm_article=KB0383441

To set up GPU on windows docker:
1. upgrade nvidia driver
2. wsl --install in administrator mode
3. enable wsl2 and ubuntu in docker
4. follow the instruction: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#install-guide

More reference: https://medium.com/@KNuggies/tensorflow-with-gpu-on-windows-with-wsl-and-docker-75fb2edd571f