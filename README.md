# PU-AT
This repository contains the implementation for the paper "计算机与数字工程: 基于Transformer的点云上采样方法"  
# Getting Started
1、Clone the repository:  
```python  
git clone https://github.com/Vencoders/PU-AT.git
```
Installation instructions for Ubuntu 18.04:

Make sure CUDA and cuDNN are installed. Only this configurations has been tested:

Python 3.7.11, Pytorch 1.6.0
Follow Pytorch installation procedure. Note that the version of cudatoolkit must be strictly consistent with the version of CUDA

2、Install KNN_cuda.
```python
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
3、Install Pointnet2_ops
```python
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git/#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```
4、Install emd
```python
cd emd_module
python setup.py install
```
5、Install h5py tensorboard
```python
conda install h5py
conda install tensorboard
```
6、Train the model: First, you need to download the training patches in HDF5 format from [GoogleDrive](https://drive.google.com/drive/folders/1Mam85gXD9DTamltacgv8ZznSyDBbBovv) and put it in folder data. Then run:
```python
python main.py --log_dir log/PU-AT
```
7、Evaluate the model: 
```python
python main.py --phase test --log_dir log/PU-AT --checkpoint_path model_best.pth.tar
```
You will see the input and output results in the folder log/PU-AT.

The training and testing mesh files can be downloaded from GoogleDrive.

Evaluation code
We provide the evaluation code. In order to use it, you need to install the CGAL library. Please refer this link and PU-Net to install this library. Then:


Related Repositories
The original code framework is rendered from "PUGAN_pytorch". It is developed by Haolin Liu at The Chinese University of HongKong.

The original code of emd is rendered from "MSN". It is developed by Liu Minghua at The University of California, San Diego.

The original code of chamfer3D is rendered from "chamferDistancePytorch". It is developed by ThibaultGROUEIX.

The original code of helper is rendered from "Self-supervised Sparse-to-Dense: Self-supervised Depth Completion from LiDAR and Monocular Camera". It is developed by Fangchang Ma, Guilherme Venturelli Cavalheiro, and Sertac Karaman at MIT.
