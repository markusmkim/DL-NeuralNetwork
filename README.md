# A general purpose neural network built from scratch
### with the use of Python and NumPy only:heavy_exclamation_mark:
#### The following features are included:
- Two types of layers: 
  - **Dense**: Any number of layers - any number of neurons in each layer
  - **Convolutional**: Any number of input channels - any number of output channels - any kernel shape - any stride - any mode
- 4 different activation functions: **Sigmoid - Tanh - Relu - Linear**
- Optional **softmax** activation for output layer
- Two different loss functions: **Mean squared error - Cross entropy**
- Two options for weight regularization: **L1 - L2**
- Any number of epochs - any mini batch size
- **SGD** is used, no other optimizers available
- Data generator for generating images to run classification on


## Performance
The system was tested on image classification.
The dataset was generated from the image generator.
It consisted of 2000 images (16x16 pixels) divided into 4 different classes (Horizontal edges, vertical edges, crosse and rectangle).
The dataset was split as follows:
- Training size: 1400 (70 %)
- Validation size: 300 (15 %)
- Test size: 300 (15 %)

The figure below shows some examples. Note that some noise is added to each image.

Horizontal edges | Vertical edges | Cross | Rectangle
------------ | ------------- | ------------- | -------------
![hor0](/data/examples/fig-0.png) ![hor1](/data/examples/fig-4.png)  | ![ver0](/data/examples/fig-1.png) ![ver1](/data/examples/fig-5.png) | ![cross0](/data/examples/fig-2.png) ![cross1](/data/examples/fig-6.png) | ![rect0](/data/examples/fig-3.png) ![rect1](/data/examples/fig-7.png)

### Configuration
The following network structure was used:
- 1st layer: Convolutional (kernel-shape=(3,3), num-kernels=2, stride=1, mode=same, activation=tanh, l-rate=0.01)
- 2nd layer: Convolutional (kernel-shape=(3,3), num-kernels=3, stride=1, mode=valid, activation=tanh, l-rate=0.01)
- 3rd layer: Dense (100 neurons, activation=relu, l-rate=0.01)
- 4th layer: Dense (4 neurons, activation=softmax, l-rate=0.001)

The following learning parameters was used:
- Loss function: Cross entropy
- Mini-batch size: 16
- Epochs: 4
- Weight regularization: L2 with regularization constant = 0.05

### Results
From the figure below it is clear that the network is learning. Note that the green line at the end is included
only to showcase the loss on the test set after training, the length of the line is arbitrary.
![loss](/data/result/fig-loss.png)


Sometimes it could be interesting to visualize the kernels in the convolutional layers as well.
The figures below shows the resulting kernels after training, for the two convolutional layers.
They are visualized as Hinton diagrams [1], where
- Positive values are represented by white squares
- Negative values are represented by black squares
- The size of the square represents the magnitude of the value

![kernel1](/data/result/kernels-layer-1.png) | ![kernel2](/data/result/kernels-layer-2.png)


## References
[1] https://matplotlib.org/stable/gallery/specialty_plots/hinton_demo.html