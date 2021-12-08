# Robust Block Detector Based on Augmented AutoEncoder

## Installation (on Linux)
1. Install assimp and OpenGL:
```
sudo apt install libassimp-dev assimp-utils
sudo apt install libglfw3-dev libglfw3
```
2. Create an Anaconda environment from the file `aae.yml`:
```
conda env create -f aae.yml
```
3. Activate the environment:
```
conda activate aae
```
4. Clone the Augmented AutoEncoder repo (https://github.com/DLR-RM/AugmentedAutoencoder.git) and checkout the `multipath` branch.
5. Install the `auto_pose` package by running `pip install .` in the root directory of this repo.
6. Download the workspace `detect_blocks_aae_ws.zip` and extract into a directory of your choice.
7. Set the path to the workspace:
```
export AE_WORKSPACE_PATH=/path/to/detect_blocks_aae_ws
```

For troubleshooting see https://github.com/DLR-RM/AugmentedAutoencoder.

## Usage
For an example script see `detect_blocks/examples/example_usage.py`.

## Evaluation on a GoodAI dataset
To run evaluation on a GoodAI dataset containing directories `screenshots` and `models` and a metadata file:
```
cd detect_blocks
./eval_<num. decoders>_dec_alg_<3 or 4>.sh /path/to/dataset/root/dir
```

You can choose `num. decoders` to be 1, 4 or 16 and you can choose between decision algorithm 3 (standard threshold decision) and 4 (modified, max-similarity decision).

Launching the script with a single argument will run evaluation on the original screenshots and models specified in `metadata_files.txt` using model poses stored in the metadata files for each screenshot. For more parameters see the documentation of the script at `detect_blocks/evaluate_on_dataset.py`.

__IMPORTANT:__ similarity detector can only use triangulated models in .ply format. To convert all models in the dataset from .obj to .ply please run the script `detect_blocks/convert_obj_to_ply.py` in Blender. Make sure that the models in .obj files have all faces triangulated before converting or remove the non-triangulated models from `metadata_files.txt` since they can't be used with the similarity detector.

## List of commands to reproduce results

Please go through the installation steps above first!

### Generating training datasets
Clone the BlenderProc repo (e.g. from `https://github.com/DLR-RM/BlenderProc`) and checkout the commit `4c7966dad01f1a16b4a5365a3d385b0d2b2b845d`.  Create a directory for image output (e.g. `/tmp/train_datasets`) and run the following:
```
cd data_generation
./gen_train_dataset_<A or B>.sh /path/to/goodai/dataset /path/to/BlenderProc /path/to/output_dir
```

### Training on generated datasets
To train the MPAAE, run:
```
python detect_blocks/load_dataset.py train/dset_<A or B>_<num. dec>_dec /path/to/opengl/images /path/to/blenderproc/output
ae_train train/dset_<A or B>_<num. dec>_dec
```
where `num. dec` can be 1, 4 or 16 for dataset A and 1 or 20 for dataset B.  If you generated the datasets with the scripts above, the OpenGL image dir is `script_output_dir/opengl` (e.g. `/tmp/train_datasets/opengl`) and BlenderProc output dir is `script_output_dir/blenderproc/output` (e.g. `/tmp/train_datasets/blenderproc/output`).

To train the Siamese networks, run:
```
cd siamese_networks
./train_<contrastive/triplet>_<A/B>.sh /path/to/opengl/images /path/to/blenderproc/output
```

### Running evaluations

See the "Evaluation on a GoodAI dataset" section.
