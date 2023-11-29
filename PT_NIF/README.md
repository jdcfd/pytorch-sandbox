# Neural Implicif Flow in Pytorch
This code tries to reproduce the NIF paper results using PyTorch Lightning.

Original implementation in Tensowflow:

https://github.com/pswpswpsw/paper-nif

## Model architecture
This model is a type of "hyper-network", where a neural network (ParamNet) is used to output the parameters of a second network (ShapeNet). As such, only the parameters for the ParamNet are learned by the model. 
