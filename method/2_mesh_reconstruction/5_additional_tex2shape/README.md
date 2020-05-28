# Tex2Shape

This repository contains code corresponding to the paper [**Tex2Shape: Detailed Full Human Body Geometry from a Single Image**](https://arxiv.org/abs/1904.08645).

## Installation

Download and unpack the SMPL model from http://smpl.is.tue.mpg.de/ and link the files to the `vendor` directory.
```
cd vendor/smpl
ln -s <path_to_smpl>/smpl_webuser/*.py .
```

Download the neutral SMPL model from http://smplify.is.tue.mpg.de/ and place it in the `assets` folder.
```
cp <path_to_smplify>/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl assets/neutral_smpl.pkl
```

Download pre-trained model weights from [here](https://drive.google.com/open?id=1yl4m7rzr-F9qbBqH-NzRqUQiD5uTTW8P) and place them in the `weights` folder.

```
unzip <downloads_folder>/weights_tex2shape.zip -d weights
```



Or just download all assets, vendor and weights folder from [here](https://drive.google.com/open?id=1faFUtsuzrCBAJw9kVzpYcCoRrxDzcohm)

## Usage

We provide a run script (`run.py`) and sample data for single subject and batch processing.
The script outputs usage information when executed without parameters.

### Quick start

We provide sample scripts for both modes:

```
bash run_demo.sh
bash run_batch_demo.sh
```

## Data preparation

If you want to process your own data, some pre-processing steps are needed:

1. Crop your images to 1024x1024px.
2. Run [DensePose](http://densepose.org/) on your images.

Cropped images and DensePose IUV detections form the input to Tex2Shape. See `data` folder for sample data.

### Image requirements

The person in the image should be roughly facing the camera, should be fully visible, and cover about 70-80% of the image height.
Avoid heavy lens-distortion, small focal-legths, or uncommon viewing angles for better performance.
If multiple people are visible, make sure the IUV detections only contain the person of interest.
