Conditional AE with discriminators on feature and reconstruction, as well as a classifier

the encoded features are in range (-1, 1), the conditional label (expression class) are in one hot coding, e.g. 1, -1 , -1, ...

dataset: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset?resource=download
put the dataset in ./dataset

data are normalized into range (-1, 1), train on the whole train dataset and use validation dataset as test dataset

Docker usage:
To build: docker build . -t "fer_train"
To run: docker run -e PYTHONUNBUFFERED=1 --runtime=nvidia --gpus all "fer_train"

To set up GPU on windows docker:
1. upgrade nvidia drive
2. wsl --install in administrator mode
3. enable wsl2 and ubuntu in docker
4. follow the instruction: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#install-guide