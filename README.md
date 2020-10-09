# **Traffic Sign Recognition** 

[//]: # (Image References)

[sign_types]: ./examples/sign_types.png "Sign Types"
[data_sets]: ./examples/data_set_summary.png "Data Sets"
[validation_accuracy]: ./examples/validation_accuracy.png "Validation Accuracy"
[sample_images]: ./examples/sample_images.png "Sample Images"
[predictions]: ./examples/top_predictions.png "Top 5 Predictions"

## Data Set Loading and Examination

### Loading Data

Three separate pickled files are provided, each containing a different number of sample features and labels. Feature images are in 32 x 32 (W x H) dimension, with three color channels. Labels are integer-encoded description of the images, where we have 43 different sign types. The number of samples in each set is given below:

| Feature Set | Number of Samples |
|:---------------------:|:---------------------------------------------:|
| Training Set | 34,799 |
| Validation Set | 4,410 |
| Test Set | 12,630 |

### Visual Verification

A file (`signnames.csv`) matching id's to descriptions is given. It is read one line at a time, and a dictionary is constructed where each integer id is mapped to the sign name (`description`) and a sample image from the validation set (`sample`). This dictionay will be used later to visaulize prediction results as well. The following figure gives a graphical representation of the dictionary.

![Map from id to sign description and sample image][sign_types]

### Data Set Summary

The occurrence of each type of traffic sign in each of the feature sets is summarized as follows:

![Data Set Summary][data_sets]

Certain types of traffic sign has more image sample than others, and distributions are approximately proportional across the three sets. This unniformity can cause degradation of model quality, but I decided to use the sets as they are and see where those sets lead me.

## Design and Test a Model Architecture

### Preprocessing

The LeNet architecture shown in the lectures (and used in the lab) is used without any major modification. The only changes are: (1) output classes are now 43 instead of 10 used in classifying MNIST handwriting images, and (2) dropout is applied for regularization in three stages later in the pipeline.

The LeNet classifier takes its input in the form of 32 x 32 x 1 images, while the images in the features sets are in the shape of 32 x 32 x 3 because they are encoded in three color channels (verified above). First, we take a grayscale for each image, and then normalize the single channel to [-1, 1], which roughly sets the mean at zero.

```
X_train = (np.mean(X_train, axis=3, keepdims=True) - 128) / 128
X_valid = (np.mean(X_valid, axis=3, keepdims=True) - 128) / 128
X_test = (np.mean(X_test, axis=3, keepdims=True) - 128) / 128
```

### Model Architecture

The model is summarized below.

| Layer | Description |
|:---:|:---:|
| Input | 32 x 32 x 1 grayscale image |
| Convolution 5 x 5 | 1 x 1 stride, valid padding, outputs 28 x 28 x 6 |
| RELU | |
| Max pooling | 2 x 2 stride, outputs 14 x 14 x 6 |
| Convolution 5 x 5 | 1 x 1 stride, valid padding, outputs 10 x 10 x 16 |
| RELU | |
| Max pooling | 2 x 2 stride, outputs 5 x 5 x 16 |
| Flattening | outputs 400 |
| Dropout | keep probability 0.5 |
| Fully connected | outputs 120 |
| RELU | |
| Dropout | keep probability 0.5 |
| Fully Connected | outputs 84 |
| RELU | |
| Dropout | Keep probability 0.5 |
| Output | outputs 43 |

### Training

The model is trained by an Adam Optimizer. The training hyperparameters are summarized below:

| Paramter | Value |
|:---:|:---:|
| Epochs | 100 |
| Batch Size | 128 |
| Learning Rate | 0.0006 |

In the beginning, I tried and checked the accuracy of the LeNet classifier without any modification (learning rate of 0.001, 10 epochs). The accuracy was not very low, but unsatisfactory and too low for application in classifying real-world images. The reason was obviously underfitting (both the training set accuracy and validation set accuracy were low). Increased the epochs to 100, and this time severe overfitting was observed (training set accuracy got quite high, but validation set accuracy saturated and did not improve in later epochs.

To overcome this overfitting, initially the learning rate was lowered, but not any significant effect other than the accuracy dropping abruptly. I decided to insert dropouts in later stages in the model pipeline. Afterwards, a high (0.001) and low (0.0001) learning rates were tested, outcome was examined, and then finally set to 0.0006. The number of epochs was set to 100, since a continual (although marginal) improvement in validation accuracy was observed even in later epochs. Besides, when run on GPUs (Udacity workspace) the training did not take too long (on a personal laptop without GPU support it took unbearably long, though).

The training set accuracy and validation set accuracy were very low in the beginning, but later they got up to higher than 95% and stayed above that. The final result from epoch 100 was that we had a training set accuracy of 0.994 and a validation accuracy of 0.966. The change in both accuracies illustrated below:

![Validation Accuracy][validation_accuracy]

### Test

After the model was trained, a set of test images were run through it. As described above, the number of test images was 12,630. Test accuracy was measured at 94.6%.

## Testing against New Images

### Image Acquisition

I took sample images from the Internet; in fact, in order to obtain same-dimension German traffic sign images, I took five samples from a GitHub repository that contains the same Udacity project. (The url for this repo is 
[https://github.com/Goddard/udacity-traffic-sign-classifier](https://github.com/Goddard/udacity-traffic-sign-classifier)).

The five sample images are shown below, along with respective descriptions taken from the dictionary described above.

![Sample Images][sample_images]

### Prediction

Prediction is done by using `tensorflow.nn.top_k`, whose results will be used in the next step in analyzing the top 5 predictions for each sample image. Sample images go through the same preprocessing as in training, and then fed into the trained model.

We pick the top prediction for each sample image and check whether this prediction is equal to the label manually attached to each image (in `y_prediction`). Count the total correct prediction, divide by the number of images to calculate the accuracy.

Although this accuracy is 100%, it does not mean much. As described above, the model is expected to predict 94.6% of German traffic sign images correctly (provided that they are approximately in the same scale). The probability that this model correctly predict five out of random five images is 0.946 ** 5 = 0.758, which higher than 3/4.

### Top 5 Softmax Probabilities

The following figure shows top 5 predictions for each sample image.

![Top 5 Predictions][predictions]

For sample images 1, 3, and 5, the model is quite confident that the top prediction is correct (softmax probabilities are 100.0%, 100.0%, and 99.7%, respectively). However, in some cases, this confidence can be low or mediocre at best. For example, for sample image 4 ("No Vehicles"), the top prediction has a softmax probability of as low as 31.2%. Though this is higher than that of any other class (resulting in correct prediction), it is no wonder that this model will fail for other images showing a German traffic sign.

Enhancements is expected by

- Fine-tuning the training (and validation) set: it is desirable to have various images of traffic signs, preferably uniformly distributed to all classes. Simple graphical conversion (e.g. affinity transform) can be applied to obtain more images that can be used to train the model.
- Adjustment of learning hyperparameters: the current training setting is ad-hoc, and not very much attention is paid to iteratively tuning the parameters. Careful examination and analysis of intermediate results (and of course, good insight) will help improve the training process and the resulting prediction model.
- Architecture exploration: we tested the same settings as the original LeNet, with the only adjustement being addition of three dropouts. Empirical testing and analysis of architectural alternatives will also help improve the model.
