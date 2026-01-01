# Model Design

The oil spill detection model is a convolutional neural network
designed for image segmentation.

The model takes a 256x256 RGB satellite image as input and outputs
a multi-class segmentation mask using a softmax activation layer.

The architecture uses convolution, pooling, and upsampling layers
to learn spatial features effectively.
