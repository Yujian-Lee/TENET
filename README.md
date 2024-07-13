# TENET
## TENET: Triple-Enhancement based Graph Neural Network for Cell-cell Interaction Network Reconstruction from Spatial Transcriptomics
![image](https://github.com/Yujian-Lee/TENET/blob/main/model%20architecture.png)
# TENET Installation
Create the conda environment (Default installation path), other installation path, use -p to select your own path:
```
conda env create -f environment.yml
```
List all of the environment:
```
conda info -envs
```
To activate the environment:
```
conda activate TENET
```
If the environment cannot be installed successfully, follow the following instructions:
```
conda create --name TENET python=3.8
pip install -r requirements.txt
```
Installing pytorch torchvision torchaudio separately (check your own cuda version), my cuda version is 11.7. 
```
# To check the cuda version
nvcc -V
```
If the cuda version is different from mine, the following url and the provided .whl files should be different.
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install xxx.whl
```
These .whl files have the corresponding version, please check whether your torch version can match these packages.

# Training
To get started, run the following code:

```
cd TENET
python main.py
```
```
-m (mode; default to be "train")
-t (train-test-split ratio)
-fp, -fn (two noise ratio)
```
another hyperparamters can be modify by yourself.

# Cite
```
@article{lee2024tenet,
  title={TENET: Triple-Enhancement based Graph Neural Network for Cell-cell Interaction Network Reconstruction from Spatial Transcriptomics},
  author={Lee, Yujian and Xu, Yongqi and Gao, Peng and Chen, Jiaxing},
  journal={Journal of Molecular Biology},
  pages={168543},
  year={2024},
  publisher={Elsevier}
}
```
