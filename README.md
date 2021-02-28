# DLF (Deep Learning as a Framework)

**DLF** is a Easy to use library to get started with deep learning development. You don't need to implement popular models like ResNet, ResUNet, simple classification models, YOLO object detection models, loss functions like huber loss, contrastive loss, YOLO loss and many more.  

## Docs
We have an extensive documentation. You can find it here: [dlfdocs.iamroyakash.com](https://dlfdocs.iamroyakash.com)

## Dependencies
Built on top of Keras and tensorflow, this library provides optimized implementations for popular deep learning models and losses and other utility functions.

## Installation

We have not deployed the framework yet on PyPI, we are on the track to do so in couple of months. PyPI enables us to deploy the framework worldwide and provides a easier installation on fresh machines.

- First download/clone this repo like `git clone https://github.com/theroyakash/dlf.git`
- Now uninstall if any previous version installed `pip3 uninstall dlf`
- Now install fresh on your machine `pip3 install -e dlf`

## Alternate installation
This is easier to install this way but a bit slower in the installation time.

```bash highlight=2
pip3 uninstall dlf  # If there any previous version installed
pip3 install https://github.com/theroyakash/dlf/tarball/main
```