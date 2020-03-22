# SMPL AMC Imitator

**NOTE**: the code style is awful and I don't have time to prettify it. If anyone is really going to use this, please open an issue and I'll response as soon as possible.

For a given AMC/ASF motion sequence, we transfer the motion to SMPL model, and generate a corresponding 3D SMPL sequence.

This work is based on [my implmentation](https://github.com/CalciferZh/SMPL) of [SMPL model](http://smpl.is.tue.mpg.de/) and [my implementation](https://github.com/CalciferZh/AMCParser) of AMC/ASF parser.

## Demo

### Skeleton (left: SMPL target, right: ASF/AMC source)
![Skeleton Demo](./demo_skeleton.gif)

### Skinned Model
![Skinned Demo](./demo_skinned.gif)

## Usage

### Quick Demo

Run `python 3Dviewer.py` to see demo.

Also, run `python batch.py` to extract all poses into `./pose/` from `./data/`.


### Step by Step Tutorial

1. Use `reader.parse_asf()` to extract skeleton definition from `.asf` file: `joints = reader.parse_asf(asf_path)`

2. Use `reader.parse_amc()` to extract motion sequence from `.amc` file: `motions = reader.parse_amc(amc_path)`

3. Construct a `smpl_np.SMPLModel` object: `smpl = SMPLModel(smpl_model_path)`

4. Construct a `imitator.Imitator` object: `imit = Imitator(joints, smpl)`

5. Use `imitator.Imitator.imitate` to manipulate SMPL model to some pose: `imit.imitate(motions[frame_index])`

6. Use `smpl_np.output_mesh` to get `.obj` file: `imit.smpl.output_mesh(output_path)`

In step 5, the `SMPLModel` inside `Imitator` is set to the same pose as `motions[frame_idx]`.

For any questions, feel free to open an issue.


## Challenge
The skeleton of SMPL is a little bit different from CMU MoCap Dataset's. In this implementation, we only process femur and tibia and ignore other differences. We first pose SMPL skeleton (specifically legs) to be in the same pose with ASF defination. After that, we extract rotation matrices from AMC files and apply them to the aligned SMPL model.

Feel free to [contact me](mailto:calciferzh@outlook.com) for more details.
