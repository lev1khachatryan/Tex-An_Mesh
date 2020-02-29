## Environment

Python (both 2 and 3)
Numpy
Chumpy
TensorFlow 1

## Preprocess

Requires the SMPL model (`.pkl` file format):
```
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz && tar -xf models.tar.gz
```

The downloaded SMPL model contains `chumpy` arrays. `preprocess.py` (adapted from https://github.com/CalciferZh/SMPL) converts these to `numpy` arrays.
```
python2 preprocess.py <path-to-neutral-SMPL-model>
```

## Run

```
python3 test_runner.py <path-to-SMPL-model>
```
