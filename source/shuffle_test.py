import tensorflow as tf

def printDs(ds):
    print('---------------------')
    for example in ds.take(ds.cardinality()):
        print(example)

data = list(range(5000))
ds = tf.data.Dataset.from_tensor_slices(data)

ds = ds.shuffle(5000//3, reshuffle_each_iteration=True)

ds = ds.batch(64)

ds = ds.shuffle(10, reshuffle_each_iteration=True)

for _ in range(50):
    printDs(ds)
