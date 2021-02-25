# Residual UNets
Residual nets provides robust architectural stabilities for very very deep neural network and UNets are the one provides infrastructure for creating image segmentation masks. Together these two provides very good segmentation performance over the normal UNets. Here is a research paper describing [that](https://arxiv.org/abs/1904.00592).

So we've provided a simple implementation for Residual UNets in keras to get you started with segmentation problems.

## Usage
As always, start with importing the model from model zoo
```python
from dlf.unets import ResidualUNet
import tensorflow as tf
```

Then we need dice loss and dice coefficient for training the model on custom datasets which we've also made keras compatible and available with a single import

```python
from dlf.losses import dice_loss, dice_coef
```

Now create the model object
```python
residualUNet = ResidualUNet(128)
```
Here 128 is the input image size, you can change it however you like.

Now build the model object
```python
model = residualUNet.make_model()
```

Define optimizer and start training
```python
adam = tf.keras.optimizers.Adam()
model.compile(optimizer=adam, loss=dice_loss, metrics=[dice_coef])

model.summary()
```

Check how the model looks like
```python
# Draw the model on a image
tf.keras.utils.plot_model(model, to_file='residualunet.png', show_shapes=True)
```

See the model source code [here](https://github.com/theroyakash/dlf/blob/master/dlf/unets/resunet.py).