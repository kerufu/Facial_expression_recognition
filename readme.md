To run: docker build . -t "fer_train"


Conditional AE with discriminators on feature and reconstruction, as well as a classifier

the encoded features are in range (-1, 1), the conditional label (expression class) are in one hot coding, e.g. 1, -1 , -1, ...

dataset: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset?resource=download
put the dataset in ./dataset

data are normalized into range (-1, 1), train on the whole train dataset and use validation dataset as test dataset