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
        print("load data from raw")
        data = []
        condition_label = []
        one_hot_coding_label = []
        for ec_index in range(setting.num_classes):
            for path in glob.iglob(dataset_path+"/"+setting.expression_classes[ec_index]+"/*.jpg"):

                img = cv2.imread(path, 0)
                data.append(img)

                condition = [-1] * setting.num_classes
                condition[ec_index] = 1
                condition_label.append(condition)
                
                one_hot_coding = [0] * setting.num_classes
                one_hot_coding[ec_index] = 1
                one_hot_coding_label.append(one_hot_coding)

        data = np.array(data)
        data = data / 127.5 - 1
        data = data[:, :, :, np.newaxis]

        condition_label = np.array(condition_label)

        one_hot_coding_label = np.array(one_hot_coding_label)
        
        dataset = tf.data.Dataset.from_tensor_slices({
            "data": data,
            "condition_label": condition_label,
            "one_hot_coding_label": one_hot_coding_label
        })

        dataset = dataset.batch(setting.batch_size)
        # bug of tensorflow dataset, cannot save reshuffle_each_iteration
        # dataset = dataset.shuffle(setting.batch_size, reshuffle_each_iteration=True) 

        dataset.save(processed_dataset_path)

        return dataset