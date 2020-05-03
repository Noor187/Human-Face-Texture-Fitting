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
git clone https://github.com/Noor187/Human-Face-Texture-Fitting.git
cd Human-Face-Texture-Fitting
```
  
### 2. Run using command prompt
Run the `Masking.py` using command prompt from the `Human-Face-Texture-Fitting` directory. The input to this code is the output of the DDFA model. The input file is to be in .obj file format.
```
python Masking.py -i your_input_name.obj
```

The code will write three mesh files in the same directory as the input file. 

  ##### 1. `ConfidenceMask.obj` is the visualization of the confidence weights per vertex. These weights are used for interpolation of DDFA texture (Input texture) and the predicted texture `your_input_name_BFMTexture.obj`. 

  ##### 2. `your_input_name_BFMTexture.obj` is the texture from the BFM eigeanspace that is predicted during the code based on the Input texture.

  ##### 3. `your_input_name_FinalOutput.obj` is the actual output file of texture fitting i.e. B3 section of the paper. It is obtained by interpolating the Input texture and BFM predicted texture using the confidence weights per vertex.


### 2. Test run
```
python Masking.py -i Meshes/Test1.obj
```
It will write three files Test1_FinalOutput.obj, Test1_BFMTexture.obj and ConfidenceMask.obj in the Meshes folder.


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
