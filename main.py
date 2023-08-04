from model_worker import model_worker
from dataset_worker import dataset_worker

epoch = input("Please enter number of epoch: ")
print("You entered: " + epoch)
mw = model_worker()
dw = dataset_worker()
mw.train(int(epoch), dw.train_dataset, dw.validation_dataset)2
