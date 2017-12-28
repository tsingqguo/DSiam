# DSiam
Learning Dynamic Siamese Network for Visual Object Tracking

This work tries to equip a very fast deep tracker, i.e. SiamFC, with two online fast transformations, i.e. target variation transformation and background suppression transformation, which makes SiamFC adapt target changes while excluding background interferences. The matconvnet tool should be download and added to the corresponding fold. Please refer to http://www.vlfeat.org/matconvnet/ for details. In this version, two pretained networks can be used, i.e.vgg19 and SiamFC. The vgg19 can be downloaded from http://www.vlfeat.org/matconvnet/pretrained/. The SiamFC can be downloaded from https://github.com/bertinetto/siamese-fc. Then , put these networks into the 'model' fold.  

If you are interested in this work and use this code, please consider to cite:

Qing Guo, Wei Feng, Ce Zhou, Rui Huang, Liang Wan, Song Wang. Learning Dynamic Siamese Network for Visual Object Tracking.
In ICCV 2017.
