## Environment

Python (both 2 and 3)
Numpy
Chumpy
TensorFlow 1

## 1. Preprocess

Requires the SMPL model (`.pkl` file format):
```
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz && tar -xf models.tar.gz
```

The downloaded SMPL model contains `chumpy` arrays. `preprocess.py` (adapted from https://github.com/CalciferZh/SMPL) converts these to `numpy` arrays. Running it requires the `chumpy` library which is provided with the SMPL model (http://smpl.is.tue.mpg.de/).
```
python2 preprocess.py <path-to-neutral-SMPL-model>
```

## 2. Run

```
python3 test_runner.py <path-to-SMPL-model>
```
