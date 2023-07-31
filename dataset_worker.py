import glob
import os

import cv2
import numpy as np
import tensorflow as tf

import setting
class dataset_worker():
    def __init__(self) -> None:
        try:
            self.train_dataset = tf.data.Dataset.load(setting.processed_train_dataset_path)
        except:
            self.train_dataset = self.load_dataset(setting.train_dataset_path, setting.processed_train_dataset_path)

        try:
            self.validation_dataset = tf.data.Dataset.load(setting.processed_validation_dataset_path)
        except:
            self.validation_dataset = self.load_dataset(setting.validation_dataset_path, setting.processed_validation_dataset_path)
        
        
    def load_dataset(self, dataset_path, processed_dataset_path):
        data = []
        label = []
        label_one_hot_coding = []
        for ec_index in range(setting.num_classes):
            for path in glob.iglob(dataset_path+"/"+setting.expression_classes[ec_index]+"/*.jpg"):

                img = cv2.imread(path, 0)
                data.append(img)

                label.append(ec_index)

                one_hot_coding = [-1] * setting.num_classes
                one_hot_coding[ec_index] = 1
                label_one_hot_coding.append(one_hot_coding)

        data = np.array(data)
        data = data / 127.5 - 1

        label = np.array(label)

        label_one_hot_coding = np.array(label_one_hot_coding)
        
        dataset = tf.data.Dataset.from_tensor_slices({
            "data": data,
            "label": label,
            "label_one_hot_coding": label_one_hot_coding
        })

        dataset.save(processed_dataset_path)

        return dataset