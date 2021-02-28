# A Basic MNIST Network
We provide you a basic MNIST like network to get started with your own dataset. With a single import you can start building a basic network with required output dense units. With a fresh run on a GPU this model can give you around 99.29% testing accuracy on a MNIST Problem (99.7% is the global highest).

To build the basic network you need to import that, you can do that with the following command
```python
from dlf.classification import BasicNetwork
```

Now create the model object like this
```python
model = BasicNetwork(num_classes = 10)
```

Now let's train MNIST with this model

```python
# Setting up the dataset
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds


(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
```

Now it's time to start the training

```python

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)
```