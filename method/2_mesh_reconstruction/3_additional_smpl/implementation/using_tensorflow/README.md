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

Or, pretraind SMPL model can be found [here](https://drive.google.com/open?id=1pQ-U2p-1hXmK07bS5OUbyqrv_qdke_kO)

```
python2 preprocess.py <path-to-neutral-SMPL-model>
```

## Run

```
python3 test_runner.py <path-to-SMPL-model>
```
