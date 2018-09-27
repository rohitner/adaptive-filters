A single input–single output parameter varying ARX model is simulated as follows:

[![](http://latex.codecogs.com/svg.latex?y_t%20%3D%20a_0%20+%20a_1%20y_%7Bt-1%7D%20+%20a_2%20u_%7Bt%7D%20+%20a_3%20u_%7Bt-1%7D%20+%20%7B%5Cepsilon%7D_t)](#plots)

Here the input [![](http://latex.codecogs.com/svg.latex?%5Cinline%20u_t)]() is a uniformly distributed random number in the interval [![](http://latex.codecogs.com/svg.latex?%5Cinline%20%5B-2%2C2%5D)](), [![](http://latex.codecogs.com/svg.latex?%5Cinline%20a_0)](), [![](http://latex.codecogs.com/svg.latex?%5Cinline%20a_1)](), [![](http://latex.codecogs.com/svg.latex?%5Cinline%20a_2)](), and [![](http://latex.codecogs.com/svg.latex?%5Cinline%20a_3)]() are the ARX model parameters, and [![](http://latex.codecogs.com/svg.latex?%5Cinline%20%5Cepsilon_t)]() is zero mean white noise. For the training set, a total of 600 samples are generated, and the ARX parameters are varied with a combination of ramp and step changes as shown in the table.

![ARXTable](../arx.png)

The least squares solution for the first 20 points is used as the initial value of the ARX parameters. Finally, outliers are added to the output to observe how the different algorithms perform in the presence of extreme values.