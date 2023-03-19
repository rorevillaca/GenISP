<h1 style="text-align: center;">GenISP: Neural ISP for Low Light Machine Cognition</h1>

<h3 style="text-align: center;">A reproduction by R. Revilla Llaca, V. Costa and P. Jain</h2>

<br><br>


## Contents
- [Contents](#contents)
- [Motivation](#motivation)
- [Introduction](#introduction)
- [Method](#method)
  - [Method Overview](#method-overview)
  - [Short Overview of RAW Files](#short-overview-of-raw-files)
  - [Packing](#packing)
  - [Averaging Greens](#averaging-greens)
  - [CST Matrix](#cst-matrix)
  - [Conv WB](#conv-wb)
  - [Resizing (Bilinear Interpolation)](#resizing-bilinear-interpolation)
  - [Architecture](#architecture)
  - [MLP](#mlp)
  - [Conv CC](#conv-cc)
  - [Shallow ConvNet](#shallow-convnet)
  - [Detector](#detector)
- [Results](#results)
- [Discussion](#discussion)
***
<br>

## Motivation

Address the value of doing a reproduction

## Introduction
## Method
### Method Overview
### Short Overview of RAW Files
RAW is a class of computer files containing all the information available to a camera sensor when the shutter is clicked. This information is usually split into two files: the image data, that represents the light intensity and color of a scene in pixels, and the metadata containing the contextual information of the image. 


As opposed to processed filetypes (such as JPEG or PNG), RAW files are larger in size and are not directly recognizable to the human eye, but allow for a more granular and tehnical manipulation of the information. In order to process the image, both the metadata and pixel information must be combined. 

The structure of a RAW file changes from manufacturer to manufacturer, and even between camera models. For this explanation we will focus on .ARW files, which are the RAW files produced by Sony Alpha cameras, as this is the format for the provided training data. The pixel information in .ARW files is stored in an m-by-n array, where each pixel contains the information from a unique color channel (red, green or blue). The image metadata contains information about the pattern, which will help us process the image as described in the article:

```python
with rawpy.imread('DSC01375.ARW') as raw:
    print(f'File pattern: {raw.raw_pattern.tolist()}')
    print('Color description: ', raw.color_desc)
```
This outputs:
```
File pattern: [[0, 1], [3, 2]]
Color description:  b'RGBG'
```
As can be seen, the information for each pixel is contained in contiguous 2-by-2 arrays Furthermore, the sensor for the camera contains twice as many pixels for green than for blue or red. This is common in Bayer sensors because the human eye is more sensitive to variations in shades of green. The green color values are stored in positions "1" and "3" (that is, the upper right and lower left positions of each array). Red and blue values are stored in the "0" (upper left) and "2" (lower right) positions respectively. 


### Packing

The goal of packing (also known as demosaicing) is to extract the information for each color channel into its own array. The dimensions of the resulting arrays will be half of those in the original (RAW) array. The packing procedure for the red channel is illustrated below. 

![](g7916.png)

To pack the RAW images, we can simply subset the values according to their index and store them in an array for each channel.

### Averaging Greens

In the paper the green channels are averaged linearly, resulting in the standard RGB channels. This process is shown below: 

![](g7985.png)

### CST Matrix

As a last preprocessing step, the image is converted to the XYZ color space. As the paper explains, converting the images to this device-independent color space ensures that the method generalizes to unseen camera sensors. The transformation is performed by applying the CST matrix, which is included within the image metadata. 

To apply the CST matrix, it must be multiplied by the values for every pixel in the image. An example is provided below:
 

### Conv WB

### Resizing (Bilinear Interpolation)
### Architecture
### MLP

### Conv CC
### Shallow ConvNet
### Detector
## Results
## Discussion
State if your reproduction results uphold the main conclusions of the paper
