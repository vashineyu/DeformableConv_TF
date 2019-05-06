# DeformableConv_TF
Implementation of DeformableConv Layer in Tensorflow-Keras
modified from https://github.com/DHZS/tf-deformable-conv-layer

# Main requirements
tensorflow v1.13  
yacs  
imgaug  

# Usage
We use yacs.config to control parameter setting  
`python run.py` # run with default. 3 layers of standard convolution, with cpu mode  
`python run.py MODEL.USE_DEFORMABLE_CONV True SYSTEM.DEVICES "[0]"` make standard conv as deformable conv, use gpu (id=0)
