import os
os.chdir("..")

from model_worker import model_worker
from dataset_worker import dataset_worker

epoch = input("Please enter number of epoch: ")
print("You entered: " + epoch)
aei = input("Please enter number of auto encoder iteration: ")
print("You entered: " + aei)
edi = input("Please enter number of encoder discriminator iteration: ")
print("You entered: " + edi)
ddi = input("Please enter number of decoder discriminator iteration: ")
print("You entered: " + ddi)
ci = input("Please enter number of classifier iteration: ")
print("You entered: " + ci)
mw = model_worker(int(aei), int(edi), int(ddi), int(ci))
dw = dataset_worker()
mw.train(int(epoch), dw.train_dataset, dw.validation_dataset)
