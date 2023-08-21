import tensorflow as tf

def printDs(ds):
    print('---------------------')
    for example in ds.take(ds.cardinality()):
        print(example)

data = list(range(50))
ds = tf.data.Dataset.from_tensor_slices(data)

ds = ds.shuffle(10, reshuffle_each_iteration=True)

ds = ds.batch(10)

printDs(ds)

printDs(ds)