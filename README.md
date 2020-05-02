# Synthesizing Normalized Faces From Facial Identity Features (Additional Results B.3 Section Implementation Only)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![GitHub issues](https://img.shields.io/github/issues/nabeel3133/3D-texture-fitting.svg)
![GitHub repo size](https://img.shields.io/github/repo-size/nabeel3133/3D-texture-fitting?style=plastic)

<p align="center"> 
<img style="display:inline;" width="250" height="300"; src="/images/Input.gif">
<img style="display:inline;" width="250" height="300"; src="/images/Input.gif">
<img style="display:inline;" width="250" height="300"; src="/images/Output.gif">
</p>

This repository provides a Python implementation of the CVPR 2017 Paper - Synthesizing Normalized Faces From Facial Identity Features. Code covers only the Fitting Texture part that is mentioned in Additional Results Section B.3 of the paper.

## Paper
[Synthesizing Normalized Faces From Facial Identity Features](https://arxiv.org/pdf/1701.04851.pdf)

## Dependencies
* [Python >= 3.5](https://www.python.org/downloads/release/python-352/)
  - Numpy-> ```pip install numpy```
  - Scipy -> ```pip install scipy```
  - Math->```pip install maths```
  
## Usage
### 1. Cloning the repository
```
https://github.com/Noor187/Human-Face-Texture-Fitting.git
cd Human-Face-Texture-Fitting
```
  
### 2. Test run
Run the `Masking.py` with obj output from ddfa as input
```
python Masking.py -o ./Inputs/your_input_name
```

The input to this code is the output of the DDFA model. The input file is to be in .obj file format.

Output file (.obj) will be saved in `Human-Face-Texture-Fitting/Output` folder with the name `Masked_your_input_name.obj` which can be rendered by Meshlab or Microsoft 3D Builder.

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
