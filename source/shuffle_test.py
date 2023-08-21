import tensorflow as tf

def printDs(ds):
    print('---------------------')
    for example in ds.take(ds.cardinality()):
        print(example)

data = list(range(500))
ds = tf.data.Dataset.from_tensor_slices(data)

ds = ds.shuffle(500//3, reshuffle_each_iteration=True).batch(64)

for _ in range(50):
    printDs(ds)
