# AerialSeg
AerialSeg is a collection of algorithm pipelines for segmentation of aerial imagery implemented by PyTorch, which consists of following elements with characteristics.

- Dataset & Dataloader: Original ISPRS Potsdam dataset is supported and there is no need to divide large images into smaller ones before training anymore. AerialSeg adapts modified random sampling mechanism to fully make use of context information without waste by division.
- Data augmentation: AerialSeg offers a set of data augmentation transforms considering the unique characteristics of aerial imgery, such as rotation invariance.
- Loss function: The distribution of classes in aerial images is usually imbalanced so loss function should be sensitive to classes with a small proportion.
- Evaluation & Monitoring: AerialSeg provides 4 metrics of evaluation, namely Acc, Acc per class, mIoU and FWIoU. TensorBoardX is applied to keep track with training process.

## Dataset & Dataloader

AerialSeg allows direct use of original VHR dataset without massive preprocess (for example, dividing large patches into smaller ones) and a set of data augmentation transforms especially for aerial TOP images.

The original random cropping transform provided by torchvision is to randomly choose a coordinate origin so that the sampling is not pixel-wise. Experiment results are shown below.![](https://github.com/QT-Zhu/AerialSeg/blob/master/images/random_1.png)

This sampling mechanism is slightly modified so that pixels nearby the image margin get compensated. New results are shown below.![](https://github.com/QT-Zhu/AerialSeg/blob/master/images/random_2.png)

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
3. Pay attention to the relationship among version of driver, CUDA and PyTorch. Please refer to the document of [NVIDIA](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) and official site of [PyTorch](https://pytorch.org).

## Configuration

1. Strongly recommend to use Anaconda to configure the environment by `conda create -n AerialSeg python=3.7.5`.
2. For macOS, `conda install pytorch torchvision -c pytorch`, and for Ubuntu with CUDA, `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch` to install PyTorch and torchvision.
3. Install sklearn by `conda install scikit-learn` and install TensorBoardX by `conda install -c conda-forge tensorboardx`.
4. Hopefully, no other major site packages are required.

## Todo

- [x] Support FCN
- [x] Support DeepLabV3
- [x] Support Lovász-Softmax loss
- [ ] Support DeepLabV3+
- [ ] Support CARAFE upsampling module
- [ ] Provide a simple GUI