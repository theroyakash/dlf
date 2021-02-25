# Huber loss

The Huber loss function describes the penalty incurred by an estimation procedure f. Huber (1964) defines the loss function piecewise by

![HuberLoss](/huberloss.svg)

## Using Huber Loss
First Import Huber Loss from DLF losses, and make an object out of it.

```python
from dlf.losses import HuberLoss
huberloss = HuberLoss(threshold=0.9)
```
See What the threshold function doing in the loss function at [wikipedia](https://en.wikipedia.org/wiki/Huber_loss).

Next Use that in the `model.compile(...)` method when training the network like this
```python highlight=2
# Several Model code later ...
model.compile(optimizer='adam', loss=huberloss)
model.fit(Xs, ys, ...)
model.predict(...)
```