<h1 align="center">
  <br>
  <a href="https://github.com/theroyakash/dlf"><img src="https://i.imgur.com/hoWNKR5.png" alt="AKDSFramework" width="800"></a>
  <br>
  Deep Learning as a framework
  <br>
</h1>

<h4 align="center">PyTorch based python deep learning package that provides all basic implementations for popular deep learning and machine learning algorithms.</h4>

Our Package will allow user to quickly setup and use popular deep learning algorithms and not to worry about finding implementation online or doing that manually.

# What?
We didn't develop a framework from scratch, but we provide pre-built models on top of built from scratch models so that people just could import and use them.

See how easy it is to use YOLO for object detections. `MOCK EXAMPLE, API CAN CHANGE DURING THE PRODUCT LIFE CYCLE`

```python
import dlf

yolov1 = dlf.YOLO()
loss = dlf.losses.YOLOLoss()

history = yolov1.train(withLoss=loss)
```