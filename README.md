
# Optimizers for backpropagation algorithms and autoencoders
In this repository with the help of the two tasks, I have explained different optimizers for backpropagation algorithms and have also explained about the uses and autoencoders and the uses of noise in the data. Problem statement is given below :- <br>
Task 1 is to train the fully connected neural network (FCNN) using different optimizers for the backpropagation algorithm and compare the number 
of epochs that it takes for convergence along with their classification performance. Task 2 includes building an autoencoder to obtain the hidden representation and use it for classification.
<ol>
  <li><strong>Task based on different optimizers:-</strong></li>
  <ol>
   a. Develop an FCNN with 3 hidden layers. Use cross-entropy loss. Experiment with
   different a number of nodes in each of the layers. Train each of the architectures using 
   (a) stochastic gradient descent (SGD) algorithm - (batch_size=1), (b) batch gradient 
   descent algorithm (vanilla gradient descent) – (batch_size=total number of training 
   examples), (c) SGD with momentum (NAG) – (batch_size=32), (d) RMSProp 
   algorithm – (batch_size=32), and (e) Adam optimizer – (batch_size=32). Use
   the difference between average error of successive epochs fall below a threshold 10-4 as 
   convergence criteria. Consider β1 = 0.9, β2 = 0.999 and ε = 10-8 for Adam optimizer. 
   Consider momentum parameter as 0.9, learning rate as 0.001 and β = 0.99 for RMSProp. 
   
   <ol>
   i. Observe the number of epochs considered for convergence for each of the 
   architectures. Tabulate and compare the number of epochs considered by each of 
   the optimizers for each architecture.<br>
   ii. Present the plots of average training error (y-axis) vs. epochs (x-axis). <br>
   iii. Give the training accuracy and validation accuracy for each of the optimizers in 
   each of the architectures.<br>
   iv. Choose the best architecture based on validation accuracy. Give the test 
   confusion matrix and test classification accuracy along with training accuracy 
   and confusion matrix for the chosen best architecture.<br>
    </ol>
   </ol>
 
</ol>

<ol>
  <strong>2. Task based on autoencoder:-</strong>
  <ol>
   a. Develop an autoencoder that learns a compressed representation of the input features for 
   a classification predictive modeling problem. Train the autoencoder using Adam 
   optimizer. Use sigmoid activation function for the nodes in all the hidden layers. Use the 
   difference between average error of successive epochs fall below a threshold 10-4 as 
   convergence criteria.
   
   <ol>
   i. Build autoencoders with one hidden layer and 3 hidden layer architectures. For 
   each architecture, experiment with a different number of neurons in hidden 
   layers including the compressed layer. Present the mean squared error (MSE) 
   i.e., the average reconstruction error for training data as well as validation data 
   for each of the architectures.<br>
   ii. Choose the best architecture for (a) encoder with one hidden layer
   (b) encoder with 3 hidden layer architectures based on validation error. Give the test reconstruction error for the chosen best architectures.<br>
   iii. Present the plots of average training reconstruction error (y-axis) vs. epochs (xaxis) for the best architecture for (a) encoder with one hidden layer and (b) 
     encoder with 3 hidden layer architectures. <br>
   iv. Take one image from the training set and one image for the validation set, from 
   each of the classes, and give their reconstructed images for each of the experiments.<br>
   v. Classification using the compressed representation from the encoder with one hidden layer:<br>
   <ol>
      • Present each training data to the best encoder with one hidden layer and 
      save the output of the hidden layer (compressed layer). This gives the 
      compressed representation of training data. Similarly obtain the 
      compressed representation of validation and test data. <br>
      • Build the FCNN using Adam optimizer for classification. Experiment
      with the different number of hidden layers and the different number of 
      neurons in each hidden layer. Select the best architecture based on 
      validation accuracy. Report the train, validation & test accuracy along 
      with the confusion matrix and compare the results with the best result 
      from Task 1. <br>
    </ol>
    vi. Classification using the compressed representation from the encoder with 3
    hidden layers: Repeat the experiments as described in previous question(Task 2.a.iii)<br>
    vii. Weight visualization: For the best compressed representation in one hidden layer
    autoencoder, plot the inputs as images that maximally activate each of the 
    neurons of the hidden representations (plot of weights from the input layer to the 
    compressed layer).<br>
    </ol>
    b. Develop a denoising autoencoder with 20% noise and 40% noise for the best one-hidden 
    layer autoencoder architecture from Task 2.a.i. Follow the concepts discussed in the 
    class to corrupt the inputs on the fly during training. 
    <ol>
    i. Take the same images used in Task 2.a.ii. and give their reconstructed 
    images<br>
    ii. Give the classification accuracy for classification using the compressed 
    representation. Compare their performance with that of the corresponding vanilla
    autoencoder from Task 2.a.iii. <br>
    iii. Weight visualization: Plot the inputs as images that maximally activate each of 
    the neurons of the hidden representations obtained using both the denoising 
    autoencoders (plot of weights from the input layer to the compressed layer). 
    Compare it with that of the images obtained in Task 2.a.v.<br>
    </ol>
   </ol>
 
</ol>
