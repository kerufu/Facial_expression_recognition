import os
os.chdir("..")

from caae_worker import caae_worker
from wgan_worker import wgan_worker
from dataset_worker import dataset_worker

dw = dataset_worker()

model = input("Please choose model:\n 1. CAAE   2. WGAN\n")
print("You entered: " + model)

epoch = input("Please enter number of epoch: ")
print("You entered: " + epoch)

if model == "1":
    aei = input("Please enter number of auto encoder iteration: ")
    print("You entered: " + aei)
    edi = input("Please enter number of encoder discriminator iteration: ")
    print("You entered: " + edi)
    ddi = input("Please enter number of decoder discriminator iteration: ")
    print("You entered: " + ddi)
    ci = input("Please enter number of classifier iteration: ")
    print("You entered: " + ci)
    mw = caae_worker(int(aei), int(edi), int(ddi), int(ci))
    mw.train(int(epoch), dw.train_dataset, dw.validation_dataset)
elif model == "2":
    mw = wgan_worker()
    mw.train(int(epoch), dw.train_dataset, dw.validation_dataset)
