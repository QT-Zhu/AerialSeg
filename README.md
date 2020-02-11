# AerialSeg
AerialSeg is a collection of algorithm pipelines for segmentation of aerial imagery implemented by PyTorch, which consists of following elements with characteristics.

- Dataset and Dataloader: ISPRS Potsdam only now
- Data augmentation
- Special loss function
- Evaluation

## Environment

Tests are done with following environments:

### Ubuntu 16.04

- Python = 3.7.5
- PyTorch = 1.2.0
- torchvison = 0.4.0
- CUDA =10.0
- NVIDIA GPU driver = 410.78

### macOS Mojave 10.14.6

- Python = 3.7.5
- PyTorch = 1.3.0
- torchvision = 0.3.1

Note:

1. Lower versions of PyTorch may not contain implementation of AdamW optimizer.
2. Lower versions of torchvision may not contain implementation of popular segmentation models.
3. Pay attention to the relationship among version of driver, CUDA and PyTorch.

## Configuration

1. Strongly recommend to use Anaconda to configure the environment by`conda create -n AerialSeg python=3.7.5`.
2. For macOS, `conda install pytorch torchvision -c pytorch`, and for Ubuntu with CUDA, `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch` to install PyTorch and torchvision.
3. Install sklearn by `conda install scikit-learn`.
4. Hopefully, no other site packages are required.



