# HMR Code Modified for Photo Wake Up

### Requirements
- Python 2.7
- [TensorFlow](https://www.tensorflow.org/) tested on version 1.3

### Installation

#### Setup virtualenv
```
virtualenv venv_hmr
source venv_hmr/bin/activate
pip install -U pip
deactivate
source venv_hmr/bin/activate
pip install -r requirements.txt
```
#### Install TensorFlow
With GPU:
```
pip install tensorflow-gpu==1.3.0
```
Without GPU:
```
pip install tensorflow==1.3.0
```

### Demo

1. Download the pre-trained models
```
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz && tar -xf models.tar.gz
```

2. Run the demo
```
python PhotoWakeUpHMR/PhotoWakeUp.py -img_path data/coco1.png
```
