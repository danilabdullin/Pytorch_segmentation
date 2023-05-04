# Stroke Segmentation using UNet in PyTorch
This is a PyTorch implementation of a UNet model for segmenting stroke lesions from MRI images. 
# Dataset
The ISLES-2022 (Ischemic Stroke Lesion Segmentation) dataset is used for training and validation of the model. The dataset consists of MRI brain images of 100 patients, each with a stroke lesion segmentation mask
# Model Architecture
The Unet architecture is used for the segmentation task. The model consists of an encoder and a decoder. The encoder has five blocks, each consisting of two 3x3 convolution layers with batch normalization and ReLU activation. The number of feature maps in each block is doubled with each downsampling step. The decoder has five blocks as well, each consisting of an upsampling layer followed by two 3x3 convolution layers with batch normalization and ReLU activation. Skip connections are added between the corresponding encoder and decoder blocks.
# Requirements
- PyTorch (version 1.8.1 or later)
- Numpy
- Matplotlib
- nibabel

Files
UNet_stroke_segmentation.ipynb: Jupyter notebook containing the definition of the UNet model, the code for training and testing the UNet model and dataset class for loading the MRI data.


# Results
The trained UNet model achieved an average Dice coefficient of 0.85 on the test set. The segmentation results can be seen in the notebook.

Acknowledgments
This project is based on the UNet architecture proposed by Ronneberger et al. in their paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015).
