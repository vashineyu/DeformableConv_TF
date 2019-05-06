# DeformableConv_TF
Implementation of DeformableConv Layer in Tensorflow-Keras  
the deformable layer is modified from https://github.com/DHZS/tf-deformable-conv-layer  
The main modificaion is that we make the layer can support static graph (which doesn't need Eager execution)  

# Installation
Dependcies see requirements.txt. If all packages were installed, just run the code!  

# Usage
We use yacs.config to control parameter setting  
`python run.py` # run with default. 3 layers of standard convolution, with cpu mode  
`python run.py MODEL.USE_DEFORMABLE_CONV True SYSTEM.DEVICES "[0]"` make standard conv as deformable conv, use gpu (id=0)
