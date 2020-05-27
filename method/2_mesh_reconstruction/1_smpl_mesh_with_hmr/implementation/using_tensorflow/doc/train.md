## Pre-reqs

### Download required models

1. Download the mean SMPL parameters (initialization) from [here](https://drive.google.com/open?id=10R_hXb7YyJgWpRkYA8FBsHCEsLLjq2yb)

Store this inside `hmr/models/`, along with the neutral SMPL model
(`neutral_smpl_with_cocoplus_reg.pkl`).


2. Download the pre-trained resnet-50 from
[Tensorflow](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained)
```
wget http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz && tar -xf resnet_v2_50_2017_04_14.tar.gz
```

3. In `src/do_train.sh`, replace the path of `PRETRAINED` to the path of this model (`resnet_v2_50.ckpt`).

### Download datasets.
Download these datasets somewhere.

- [LSP](http://sam.johnson.io/research/lsp_dataset.zip) and [LSP extended](http://sam.johnson.io/research/lspet_dataset.zip)
- [COCO](http://cocodataset.org/#download) we used 2014 Train. You also need to
  install the [COCO API](https://github.com/cocodataset/cocoapi) for python.
- [MPII](http://human-pose.mpi-inf.mpg.de/#download)
- [MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/)
- [Human3.6M](http://vision.imar.ro/human3.6m/description.php)

~For Human3.6M, download the pre-computed tfrecords [here]().
Note that this is 11GB! I advice you do this in a directly outside of the HMR code base.~
_The distribution of pre-computed Human3.6M tfrecord ended as of April 4th 2019, following their license agreement. The distribution of the data was not permitted at any time by the license or by the copyright holders. If you have obtained the data through our link prior to this date, please note that you must follow the original [license agreement](http://vision.imar.ro/human3.6m/eula.php). Please download the
dataset directly from [their website](http://vision.imar.ro/human3.6m/description.php) and follow their [license agreement](http://vision.imar.ro/human3.6m/eula.php)._


## Mosh Data. 
We provide the MoShed data using the neutral SMPL model.
Please note that usage of this data is for [**non-comercial scientific research only**](http://mosh.is.tue.mpg.de/data_license).


[Download link to MoSh](https://drive.google.com/file/d/1b51RMzi_5DIHeYh2KNpgEs8LVaplZSRP/view?usp=sharing)

## TFRecord Generation

All the data has to be converted into TFRecords and saved to a `DATA_DIR` of
your choice.

1. Make `DATA_DIR` where you will save the tf_records. For ex:
```
mkdir ~/hmr/tf_datasets/
```

2. Edit `prepare_datasets.sh`, with paths to where you downloaded the datasets,
and set `DATA_DIR` to the path to the directory you just made.

3. From the root HMR directly (where README is), run `prepare_datasets.sh`, which calls the tfrecord conversion scripts:
```
sh prepare_datasets.sh
```

This takes a while! If there is an issue consider running line by line.

4. Move the downloaded human36m tf_records `tf_records_human36m.tar.gz` into the
`data_dir`:
```
tar -xf tf_records_human36m.tar.gz
```

5. In `do_train.sh` and/or `src/config.py`, set `DATA_DIR` to the path where you saved the
tf_records.


## Training
Finally we can start training!
A sample training script (with parameters used in the paper) is in
`do_train.sh`.

Update the path to  in the beginning of this script and run:
```
sh do_train.sh
```

The training write to a log directory that you can specify.
Setup tensorboard to this directory to monitor the training progress like so:
![Teaser Image](https://akanazawa.github.io/hmr/resources/images/tboard_ex.png)

It's important to visually monitor the training! Make sure that the images
loaded look right.