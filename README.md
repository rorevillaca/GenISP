<h1 style="text-align: center;">GenISP: Neural ISP for Low Light Machine Cognition</h1>

<h3 style="text-align: center;">A reproduction by R. Revilla Llaca, V. Costa and P. Jain</h2>

<br><br>


## **Contents**
- [**Contents**](#contents)
- [**Introduction**](#introduction)
- [**Method Overview**](#method-overview)
- [**Preprocessing pipeline**](#preprocessing-pipeline)
  - [**Short Overview of RAW Files**](#short-overview-of-raw-files)
  - [**Packing**](#packing)
  - [**Averaging Greens**](#averaging-greens)
  - [**CST Matrix**](#cst-matrix)
  - [**Preprocessing Approach**](#preprocessing-approach)
- [**Image Enhancement**](#image-enhancement)
  - [**Conv WB**](#conv-wb)
  - [**Resizing (Bilinear Interpolation)**](#resizing-bilinear-interpolation)
  - [**Architecture**](#architecture)
  - [**MLP**](#mlp)
  - [**Conv CC**](#conv-cc)
  - [**Shallow ConvNet**](#shallow-convnet)
- [**Object Detector**](#object-detector)
- [**Training the Network**](#training-the-network)
  - [**Forward Pass**](#forward-pass)
  - [**Backpropagation**](#backpropagation)
- [**Results**](#results)
- [**Discussion and Conclusion**](#discussion-and-conclusion)
***
<br>

## **Introduction**

Scientific publications should contain enough information (through explanations, diagrams, pseudocode, equations, etc) to allow reproductibility of the reported results with relatively low effort. While doing the reproduction, we underwent all sorts of challenges, including but not limited to:
* Lack of understanding of the problem
* Not enough information /details provided on the method
* No justifications provided for design decisions
* Versioning/legacy issues

Facing these challenges helped us understand what is needed to guarantee the reproductibility on the paper and the importance of correctly documenting and carefully explaining the details regarding the design and implementation of Deep Learning Models. The lessons learned in this project will be applied when performing our own research in the future. 


This blog post is a detailed review and reproduction log of the method utilized in [GenISP: Neural ISP for Low Light Machine Cognition](https://arxiv.org/abs/2205.03688). Its objective is to comprise the theoretical knowledge and practical considerations that we required in order to implement the method from scratch in [Python](https://github.com/rorevillaca/GenISP). We also include some lessons learned along the way, and some explanations and diagrams that would have been useful during the reproduction process. By writing this blog, we encourage the idea that scientific research should be fully transparent and reproducible.

In the **Methods** section we explain each step of the model in a thorough and detailed way. We rely on textual descriptions and diagrams, and include code snippets when we consider it necessary. We also emphasize what are the input and the outputs of each step. In the **Results** section we report the performance obtained by our model, comparing it to the results from the original paper. Finally, the **Conclusions/Dicussion** section contains our results interpretation along with insights, difference sources, limitations and additional work that could further improve our reproduction. 

<br><br>

## **Method Overview**

GenISP is a neural Image Signal Processor (ISP) able to enhance RAW image files as inputs to pretrained object detectors. The method avoids fine-tuning to specific sensor data by leveraging image metadata along with the RAW image files to transform the image to optimize object detection. The method consists of a preprocessing pipeline, followed by a color processing stage carried out by two networks (ConvWB and ConvCC) and a brightness enhancement carried out by a shallow ConvNet. The method is illustrated below:

![](./blog/architecture.png)


## **Preprocessing pipeline**

### **Short Overview of RAW Files**
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


### **Packing**

The goal of packing (also known as demosaicing) is to extract the information for each color channel into its own array. The dimensions of the resulting arrays will be half of those in the original (RAW) array. The packing procedure for the red channel is illustrated below. 

![](./blog/g7916.png)

To pack the RAW images, we can simply subset the values according to their index and store them in an array for each channel.

### **Averaging Greens**

In the paper the green channels are averaged linearly, resulting in the standard RGB channels. This process is shown below: 

![](./blog/g7985.png)

### **CST Matrix**

As a last preprocessing step, the image is converted to the XYZ color space. As the paper explains, converting the images to this device-independent color space ensures that the method generalizes to unseen camera sensors. The transformation is performed by applying the CST matrix, which is included within the image metadata. 

To apply the CST matrix, it must be multiplied by the values for every pixel in the image. In order to do this, only the first three rows (corresponding to the R,G,B channels) are considered. Below is the CST matrix for some models of Sony Alpha cameras:

```python
CTS Matrix:
[[ 1.0315 -0.439  -0.0937]
 [-0.4859  1.2734  0.2365]
 [-0.0734  0.1537  0.5997]
 [ 0.      0.      0.    ]]
```
 
### **Preprocessing Approach**

The following function is used to preprocess the images:
```python
def pack_avg_cst(raw_filename):
    with rawpy.imread(raw_filename) as raw:
        # Get raw image data
        image = np.array(raw.raw_image, dtype=np.double)

        # Get the raw pattern of the photo
        n_colors = raw.num_colors
        colors = np.frombuffer(raw.color_desc, dtype=np.byte)
        pattern = np.array(raw.raw_pattern)
        i_0 = np.where(colors[pattern] == colors[0])
        i_1 = np.where(colors[pattern] == colors[1])
        i_2 = np.where(colors[pattern] == colors[2])

        # Pack image and average green channels
        image_packed = np.empty((image.shape[0]//2, image.shape[1]//2, n_colors))
        image_packed[:, :, 0] = image[i_0[0][0]::2, i_0[1][0]::2]
        image_packed[:, :, 1]  = (image[i_1[0][0]::2, i_1[1][0]::2] + image[i_1[0][1]::2, i_1[1][1]::2]) / 2
        image_packed[:, :, 2]  = image[i_2[0][0]::2, i_2[1][0]::2]

        # CST Conversion
        cst = np.array(raw.rgb_xyz_matrix[0:3, :], dtype=np.double)        
        # Matrix-vector product for each pixel
        image_sRGB = np.einsum('ij,...j', cst, image_packed)

```


In order to reduce computational load, a subset of 100 images was selected from the provided dataset (wighting over 50GB). The preprocessing stage for these images was performed locally, and the result was uploaded to Google Collab, where the rest of the method was implemented. At this point, color correction and brightness of the images has not yet taken place, making them look dark and greenish:

![Preprocessed example](./blog/preprocessed_example.png)

___

## **Image Enhancement** 

### **Conv WB**
Conv WB is the first step of the neural network that will be trained in the process.  Some camera manufacturers implement a minimal ISP pipeline, e.g. in machine vision. ConvWB and ConvCC, can help adapt the colour space in such a case.

ConvWB focuses on the white balancing of the input image. ConvWB predicts gain for each color channel of the input and controls to adjust global illumination levels and white balance of the image. Regressed weights $w_{ii}$ of a 3 × 3 diagonal WB matrix are applied to the image as in a traditional ISP pipeline:

![](./blog/ConvWB.png)
```python
wb = WBNet() #WBNet is the network used by the authors and implemented with the same parameters
wb.to(torch.double)
output = wb(resized_image)

output = torch.mean(output, 0)
output = torch.diag(output)
```
The given code is used to take the image through the ConvWB part of the network and produce an output of size 3, this is then arranged in the form of the matrix as shown in the image. 
```python
img = prep_image
img = img.to(torch.double)
new_image_wb = torch.matmul(img, output)
new_image_wb.shape
```
The original image is now taken (size: 1836, 2752) and the matrix multiplication operation is applied to white balance the original image. 

### **Resizing (Bilinear Interpolation)**
The above matrix multiplication takes the image as its RGB components and multiplies it with the 3 weights the network produces. For the network, the images supplied are actually reduced to 256x256 resolution. This is done so as to decrease hardware load, given we are just trying to guage the white balancing and color correction aspects of the image which shouldn't be impacted much due to this size reduction. 

### **Architecture**
The architecture of the model is provided in the paper. This architecture definitely lacks details and leaves some work for us to figure out the reproducibility especially if we are going to replicate the results. The architecture as given by the authors can be seen in the following image: 


For a deeper understanding of the model, the authors have also given the architecture of the subnetworks used along with the number of neurons and the size of the kernels they have employed. This can also be seen in the following image: 

![](./blog/WBCC.png)

This is also exactly how we apply it, so as to reproduce the paper as closely as possible. The only parts where we might have differed from the original implementation is at the Instance norm and Maxpol levels where the authors haven't provided the kernal_size they use. The same is the case for the Avg Adapt Pool layer used. 

### **MLP**
The (Multi Layer Perceptrion) MLP is a standard fully connected neural network. We have implemented this using the inbuilt pytorch class MLP in the following way: 
```python

from torchvision.ops import MLP
  class WBNet(nn.Module):
    .
    .
      .
        self.mlp = MLP(in_channels = 64, 
        hidden_channels = [3])
        .
        .
        .
        h = self.mlp(h)
        return h
```
The hidden_channels arguemnt takes a list of ints which specify the number of hidden channels which in our case is 3. 

### **Conv CC**
ConvCC is the second part of the neural network. This subsection is used to color correct the image. The network is very similar to ConvWB, the only difference is the number of outputs it produces. While ConvCC just white balances every channel, this network has to color correct it. This is done by having a 3x3 matrix be multiplied with the image instead of a 3x3 DIAGONAL matrix. This is visualized in the following image as seen in the paper:
![](./blog/Convcc.png)

Overall, the network is defined in the same way as ConvWB, just with 9 outputs instead of 3. These 9 outputs are then applied to the original full sized image (with the ConvWB output) to get the final color and white balance corrected input image. 

```python
cc = CCNet()
cc.to(torch.double)
output = cc(resized_image)
output = torch.mean(output, 0)
output = output.reshape(3,3)
# new_image_wb is the white balanced full size image
img = new_image_wb

new_image_cc = torch.matmul(img, output)
# We apply permutations for comaptibility between the shallow ConvNet and CCNet
new_image_cc = new_image_cc.permute(0, 3, 1, 2)
```
### **Shallow ConvNet**
The shallow ConvNet is used to enhance the image as outputted by the WB and CC frameworks. In the words of the authors: `Color-corrected image is then enhanced by a shallow ConvNet`. This part of the network is where majority of the weights are and where the entire image is considered without size reductions. This network increases the number of outputs channels before finally converging. This might help the network look at structures in the image which otherwise are hard to find. The architecture of the model is already provided in the previous structure. 
```python
----------------------------------------------------
Layer (type)        Output Shape           Param #
====================================================
Conv2d-1          [ -1, 16, 1834, 2750]    448
InstanceNorm2d-2  [-1, 16, 1834, 2750]     0
Conv2d-3          [-1, 64, 1832, 2748]     9,280
InstanceNorm2d-4  [-1, 64, 1832, 2748]     0
Conv2d-5          [-1, 3, 1832, 2748]      195
====================================================
Total params: 9,923
Trainable params: 9,923
Non-trainable params: 0
----------------------------------------------------
Input size (MB): 57.82
Forward/backward pass size (MB): 6262.89
Params size (MB): 0.04
Estimated Total Size (MB): 6320.75
----------------------------------------------------
```
This is the part of the code where we utilize the most hardware space given the size of the image and the number of channels we inflate it to. 
The following methods is used to apply this network to the images: 

```python
final_images = ConvNet()
final_images.to(torch.double)
trained_image = final_images(new_image_cc)
```
## **Object Detector**
The objective of GenISP is to restore and enhance low-light images so that they are optimal for cognition by any pre-trained off-the-shelf object detector. In order to learn to do so, GenISP is guided during training by an object detector. The object detector of our choice is a single-stage RetinaNet, an object detector model proposed in the paper Focal Loss for Dense Object Detection by Lin, Goyal et al.

Before deciding to use RetinaNet, we first attempted to use an OpenCV implementation of YoloV3 as our object recognition model. Although this model was very easy to use, it did not use PyTorch as its backbone and we had limited ability to modify it and to access its layers. As we realized that this could potentially block us during the implementation of the backpropagation mechanism, we decided to switch the model. 

The choice of RetinaNet was motivated by the fact that it was referenced as being the object detector used to train GenISP in the GenISP paper, and because an easy to read and to modify PyTorch implementation of RetinaNet was freely available in the following GitHub repository: https://github.com/yhenon/pytorch-retinanet. This version also contains an implementation of the loss functions cited in the GenISP paper, alpha-balanced focal loss and smooth L1 loss.

Although the code in the repository is dated and does not run as is, few modifications were necessary to repair it. We tested it by first giving it sample images from the internet, on which it succesfully recognized different objects:

![](./blog/retinanet_test_output.png)

Above each bounding box is the label of the recognized object followed by its confidence score. 

***
## **Training the Network**

### **Forward Pass**

As we were not able to perform the forward pass through the Shallow ConvNet, we skipped this part of the network and connected the output of ConvCC directly to the object detector model. By displaying the resulting images along with bounding boxes drawn over them, we were able to observe that our ISP pipeline was producing images that could be used for cognition although they were still not optimal for it:

![](./blog/retinanet_processed_image.png)

***
### **Backpropagation**

After the forward pass is complete, loss can be calculated with respect to the object recognition model's output. Loss can be minimized by making increments to the parameters of the neural network (weights and biases) that go against the gradient of the loss with respect to those parameters. As the value of the loss represents classification and regression error, adjustments that lead to its diminution imply the production of images that are more suitable for the object recognition model.

### **Classification loss**

Classification loss is modeled by alpha-balanced focal loss:

![](./blog/alpha_focal_loss.png)
![](./blog/pt_focal_loss.png)

where y ∈ {±1} specifies the ground-truth class and p ∈ [0, 1] is the model’s estimated probability for the class with label y = 1.

Focal loss is a modi


***
## **Results**

***
## **Discussion and Conclusion**
State if your reproduction results uphold the main conclusions of the paper
