# Training-Iris-using-GD

### Step 1: 
Data set. Presently, the data set is sorted class-wise. It is best to randomize the
data set. You will use all 4 features during training. Split the dataset into 2 parts. One part
is for training and other testing.

### Step 2: 
Training: Use first 75 (randomized) samples to train the perceptron (minus step
function) using gradient descent. You need to document the training error. Ideally, we
would want it to be 0, but if it isn’t and your algorithm has exhausted its max_iterations
limit then there would be some cost/error which you need to document. You must also
note the cost after every 100 iterations and show how its improves gradually. A graph
would be nice with x-axis showing iterations: 100, 200 etc. and y-axis is the cost.

### Step 3: 
Testing: Once the training is complete and you have a trained model which would
be the weight vector [wo, w1, w2, w3, w4], test the classifier using the other 75 samples.
Take each sample, get it classified by the trained classifier. Count the number of
mismatches.
