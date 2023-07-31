from model_worker import model_worker
from dataset_worker import dataset_worker

mw = model_worker()
dw = dataset_worker()
mw.train(100, dw.train_dataset, dw.validation_dataset)