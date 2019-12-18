# Synthesizing Normalized Faces From Facial Identity Features (Additional Results B.3 Section Implementation Only)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![GitHub issues](https://img.shields.io/github/issues/nabeel3133/3D-texture-fitting.svg)
![GitHub repo size](https://img.shields.io/github/repo-size/nabeel3133/3D-texture-fitting?style=plastic)

<p align="center"> 
<img style="display:inline;" src="/images/test_input.gif">
<img style="display:inline;" src="/images/test_output.gif">
</p>

This repository provides a Python implementation of the CVPR 2017 Paper - Synthesizing Normalized Faces From Facial Identity Features (Additional Results B.3 Implementation Only). Only Fitting Texture part is implemented which is mentioned in Additional Results Section B.3 of the paper.

## Paper
[Synthesizing Normalized Faces From Facial Identity Features](https://arxiv.org/pdf/1701.04851.pdf)

## Dependencies
* [Python >= 3.5](https://www.python.org/downloads/release/python-352/)
  - [Numpy](https://pypi.org/project/numpy/) -> ```pip install numpy```
  - [Scipy](https://pypi.org/project/scipy/) -> ```pip install scipy```
  
## Usage
### 1. Cloning the repository
```
https://github.com/Noor187/Human-Face-Texture-Fitting.git
cd Human-Face-Texture-Fitting
```
### 2. Downloading the model
- [BFM09: Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-1-0&id=details)
  - After you have acquired BFM, extract the BaselFaceModel.tgz and go to`PublicMM1` folder, copy `01_MorphableModel.mat`, `BFM_exp_idx.mat` and paste it in `./3D-texture-fitting/configs` folder.
  
### 3. Running the code
Run the `Masking.py` with obj output from ddfa as input
```
python Masking.py -o ./Inputs/your_input_name
```
Test run
```
python Masking.py -o ./Inputs/Test
```

If you can see the following output log in terminal, you ran it successfully.
```
Predicting Ear and Neck Texture...
Predicting Ear and Neck Texture Completed
Dumped Obj file
Dumped Ply file
```
Two output files (obj and ply) will be saved in `Human-Face-Texture-Fitting/Output` folder with the name `Masked_your_input_name.obj` and `Masked_your_input_name.ply` which can be redered by Meshlab or Microsoft 3D Builder.

## Citation
If this work is useful for your research or if you use this implementation in your academic projects, please cite the following papers:
- [Synthesizing Normalized Faces From Facial Identity Features](https://arxiv.org/pdf/1701.04851.pdf)
```bibtex
@misc{cole2017synthesizing,
    title={Synthesizing Normalized Faces from Facial Identity Features},
    author={Forrester Cole and David Belanger and Dilip Krishnan and Aaron Sarna and Inbar Mosseri and William T. Freeman},
    year={2017},
    eprint={1701.04851},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
