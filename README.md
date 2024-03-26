This repository contains scripts to export and preprocess 3D data from Unity3D to the format required by [OpenScene](https://github.com/pengsongyou/openscene).

# Installation

## Unity
Download the [scene-obj-exporter](https://assetstore.unity.com/packages/tools/utilities/scene-obj-exporter-22250) Unity package and import it into your Unity project.

Replace the "OBJExporter.cs" script with the one provided in this repository.

## Python
Create a new conda environment with the required dependencies and activate it.
```bash
conda env create --name <name> --file environment.yml
conda activate <name>
```

# Usage

In the Unity Editor, select File > Export > Wavefront OBJ to export the current scene or selection to OBJ.

Make sure to check "Auto mark tex as readable" when exporting.

The file name should be "mesh.obj".

Run the preprocessing script to convert the scenes to the format required by OpenScene.
```bash
python3 scripts/preprocess.py --input <input_dir> --output <output_dir>
```
The script will process all subdirectories of the input directory and save the output in the output directory.

Sample scenes are provided in the "assets" directory.

The script supports additional options to control how the 2D and 3D data is exported, that can be viewed with the "--help" flag.
