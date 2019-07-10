# **Behavioral Cloning** 

---


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image3]: ./examples/model-retrained.png "Retrained model validation loss"
[image4]: ./examples/model-loss.png "Validation loss for the final model"
[image5]: ./examples/model.png "Architecture"
[image6]: ./examples/center-driving.gif "Center of the road driving"
[image7]: ./examples/recovery.gif "Recovery"
[image8]: ./examples/curve-driving.gif "Driving curves"
[image9]: ./examples/track2-right-of-middle.jpg "Original image"
[image10]: ./examples/track2-right-of-middle-flipped.jpg "Flipped image"
[model-image1]: ./examples/track2-middle-road.jpg "Middle of the road, track 2"
[model-image1-layer1]: ./examples/track2-middle-road-nvidia-conv2d_1.png "Middle of the road, track 2, layer 1"
[model-image1-layer2]: ./examples/track2-middle-road-nvidia-conv2d_2.png "Middle of the road, track 2, layer 2"
[model-image1-layer3]: ./examples/track2-middle-road-nvidia-conv2d_3.png "Middle of the road, track 2, layer 3"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided [simulator](https://github.com/udacity/self-driving-car-sim) and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network.
The file shows the pipeline I used for training and validating the model, and it contains a few comments to explain how
the code works.

There is an additional file [model.html](./model.html) that is the output of a Python notebook that was actually used for training.
The reason for creating a notebook in addition to a model.py file was to enable quick iteration on the model / training code
without dumping validation/training sets out of memory. Reloading the data set took a rather significant amount of time
and system memory, so keeping it preloaded was key to training quicker.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5 layers of convolution followed by 3 fully-connacted layers. The first 3 layers have a filter size of 5x5
and depths of 24, 36, and 48. The last 2 layers of convolution have a filter size of 3x3 and depths of 64. Finally the fully connected layers go from 100 to 50 to 10 units, before outputting 1 value. (`model.py` lines 95-112) 

All of the convolutional layers use RELU activation to introduce nonlinearity.

Input data is normalized in the model using a Keras lambda layer (`model.py` line 96) and cropped to remove irrelevant data.  

#### 2. Attempts to reduce overfitting in the model

The model contains 4 dropout layers in order to try to reduce overfitting (`model.py` lines 99-105) sandwiched between the convolutional layers.
The final validation/training loss numbers confirm a reasonably good fit for both the training and validation data sets.

The model was trained and validated on separate data sets (`model.py` lines 47-80, 124) with the validation set
randomly generated from the overall dataset collected. Shuffling on the training set was enabled for each training epoch.

The model was further tested by running it through the simulator and ensuring that the car could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model.py` line 114).

#### 4. Appropriate training data

My training dataset consists of images/steering angles captured from:

- Driving the track normally for ~2 laps
- Driving the track in the reverse direction for ~1 lap
- Driving around most turns of the track in the forwards direction (snippets of driving)
- Recovery maneuvers (car heading towards road boundary, correct and steer car back to the middle of the road). This was done in the forwards and backwards directions around the track.

This was repeated for both tracks to capture the greatest variety of conditions.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with the LeNet-5 architecture as described in the sample videos. (`model-old-lenet.py` lines 45-57, reproduced below without the dropout layer which I experimented with later).
```python
model = Sequential([
    Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)),
    Cropping2D(cropping=((55,25), (0,0))),
    Conv2D(filters=6, kernel_size=(5, 5), activation='relu'),
    MaxPooling2D(),
    Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(120),
    Dense(84),
    Dense(1)
])
```
I collected some simple data of driving on each track in the forwards and reverse directions (about 13,878 data points and 41,634 images),
trained the model for about 3 epochs, and ran the simulator to see how the car would behave. 
The car could drive around the track 1 without any issue.

Once I ran the car on the second track, it ran into problems navigating certain corners.

I experimented with adding dropout and changing the number of epochs, but this did not help,
and at some iteration the car started failing the first track as well as the second track.

Because the model seemed to be the limiting factor, I decided to try the alternative model described in the sample videos,
the [NVIDIA](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) model.
(`model.py` lines 95-111, reproduced below without the dropout layers which I added afterwards)

```python
model = Sequential([
    Lambda(lambda x: x / 255.0 - 0.5, output_shape=(160, 320, 3)),
    Cropping2D(cropping=((60, 25), (0, 0))),
    Conv2D(filters=24, kernel_size=(5, 5), activation='relu', strides=(2, 2)),
    Conv2D(filters=36, kernel_size=(5, 5), activation='relu', strides=(2, 2)),
    Conv2D(filters=48, kernel_size=(5, 5), activation='relu', strides=(2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    Flatten(),
    Dense(100),
    Dense(50),
    Dense(10),
    Dense(1)
])
```

I trained using the same exact parameters (same data set, 20% validation data split),
and after 3-5 epochs I was able to get the car to drive around track1 but it would still fail around certain corners of track2.

I even tried adding dropout layers at this point, but the validation loss still looked weird.

At this point I decided to redo my entire data set since I hadn't been methodical enough with data collection.

With the new data set, the car was able to drive both tracks without any failures,
but the validation loss numbers continued to look strange:

![Validation loss][image3]

The training loss was improving meaning the model was fitting the training data more closely,
while the validation loss kept getting worse with each epoch. Despite this, the car was driving well.

I started thinking about this write-up and began collecting more data about why my model was behaving the way it was.
After searching around I discovered that the `validation_split` argument in Keras' `model.fit(...)` method
did not randomize the validation data and simply chopped off the latter fraction of the data set to reserve
for the validation set.

This inspired me rewrite my dataset creation code (`model.py` lines 44-80) to randomly assign image/measurement pairs
to the validation data set, with a 0.25 probability of landing in the validation pile.

Then I modified my `model.fit(...)` statement to pass the manually generated validation data set:
```python
history_object = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), shuffle=True, epochs=10, batch_size=64, callbacks=[checkpointer])
```

I also learned how to set checkpoints and save intermediate weights when some maximum/minimum condition is achieved
(`model.py`, lines 116-122)
```python
checkpointer = ModelCheckpoint(
    filepath='best-weights.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True
)
```

This would ensure that we would save the model whenever the epoch achieved the best validation loss.

I set the model to train for 10 epochs and came back to this validation loss, which was much more reasonable and indicated
the true quality of the training data set and model:
 
![Validation loss][image4]

I reran the simulator and re-recorded the track 1 and track 2 runs (`video-track1`.mp4 and `video-track2`.mp4 respectively).

#### 2. Final Model Architecture

The final model architecture (`model.py` lines 95-111) consisted of a convolution neural network with the following layers and layer sizes:

(Output via `print(model.summary())`)
```
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_26 (Lambda)           (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_26 (Cropping2D)   (None, 75, 320, 3)        0         
_________________________________________________________________
conv2d_93 (Conv2D)           (None, 36, 158, 24)       1824      
_________________________________________________________________
dropout_44 (Dropout)         (None, 36, 158, 24)       0         
_________________________________________________________________
conv2d_94 (Conv2D)           (None, 16, 77, 36)        21636     
_________________________________________________________________
dropout_45 (Dropout)         (None, 16, 77, 36)        0         
_________________________________________________________________
conv2d_95 (Conv2D)           (None, 6, 37, 48)         43248     
_________________________________________________________________
dropout_46 (Dropout)         (None, 6, 37, 48)         0         
_________________________________________________________________
conv2d_96 (Conv2D)           (None, 4, 35, 64)         27712     
_________________________________________________________________
dropout_47 (Dropout)         (None, 4, 35, 64)         0         
_________________________________________________________________
conv2d_97 (Conv2D)           (None, 2, 33, 64)         36928     
_________________________________________________________________
flatten_26 (Flatten)         (None, 4224)              0         
_________________________________________________________________
dense_90 (Dense)             (None, 100)               422500    
_________________________________________________________________
dense_91 (Dense)             (None, 50)                5050      
_________________________________________________________________
dense_92 (Dense)             (None, 10)                510       
_________________________________________________________________
dense_93 (Dense)             (None, 1)                 11        
=================================================================
Total params: 559,419
Trainable params: 559,419
Non-trainable params: 0
_________________________________________________________________
```
Here's a visualization of the layers with their inputs/outputs using the Keras utility `plot_model`:
![Model visualization][image5]

Here is some example model output for a random new image from track 2:

![Original image][model-image1]

You can see a fairly strong representation of road boundaries in the first layer of the final model:

![Convolution layer 1][model-image1-layer1]

Layer 2 seems to pick up road surface as at least one feature:

![Convolution layer 2][model-image1-layer2]

Deeper layers get more abstract and lower-fidelity:

![Convolution layer 3][model-image1-layer3]

The `examples/` folder contains additional examples of the output of the first few layers of both the new and original models.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track 1 while driving on my best behavior:

![Center driving][image6]

I recorded the same thing driving the track in the opposite direction.

Then I recorded getting close to the edge and recovering on random corners and parts of the track:

![Recovery driving][image7]

I also recorded myself driving around most of the curves of the track:

![Curve driving][image8]

Then I repeated this process on track two in order to get additional data points.

To augment the data sat, I also flipped images horizontally to provide some more variety in the data.

For example, here is an image that has then been flipped:

![Original][image9]
![Flipped][image10]


After the collection process, I had 19,509 steering measurements and 58,527 images (center, left, right images for each steering measurement).


For the training data set, in addition to using the steering angle as the label for the center image,
I took the left and right images and created a synthetic steering angle by adding or subtracting a constant value of 0.2,
chosen rather arbitrarily as it was also given as an example value :
```python
# center image measurement
measurements.append(measurement)
# left image measurement
measurements.append(measurement + correction_factor)
# right image measurement
measurements.append(measurement - correction_factor)
``` 

Additionally, the number of images and measurements was doubled by creating a horizontally flipped copy
of each image, with the measurement inverted.

I pulled 25% of the data into a validation set at random and let the Keras `model.fit(...)` argument
`shuffle=True` handle shuffling training data. (code snippet from lines 40-80 of `model.py`)

```python
# percentage of images we want to use for validation
validation_fraction = 0.25

def validation_or_training(training, validation, data):
    is_validation = True if random.uniform(0, 1.0) < validation_fraction else False
    if is_validation:
        validation[0].append(data[0])
        validation[1].append(data[1])
    else:
        training[0].append(data[0])
        training[1].append(data[1])

...
X_train, y_train = [], []
X_valid, y_valid = [], []
for line in lines:
    measurement = float(line[3])

    center_image = plt.imread(line[0])
    validation_or_training((X_train, y_train), (X_valid, y_valid), (center_image, measurement))
    validation_or_training((X_train, y_train), (X_valid, y_valid), (cv2.flip(center_image, 1), -measurement))
``` 

The data set used for training the final model had 88,019 input/label pairs for training and
29,035 pairs for validation (see [the HTML notebook](./model.html)). 

I used an `adam` optimizer so that manually training the learning rate wasn't necessary.
With the training methodology finalized, I added a checkpointer to save the model when it had the lowest validation loss,
and let the training run for 10 epochs.

I would have experimented with more epochs but dealing with the online workspace was a pain due to the large training datasets
(transferring all the data files was not fun) and the long training times even in GPU mode meant training locally was more convenient. 
However, my laptop started falling apart after training several models, so I did not re-train for more than 10 epochs.

With the checkpointer, it would have been possible to let the model train for many more epochs, since
we would be able to capture the model at the epoch with optimal validation loss.

The lowest validation loss was achieved at epoch 10, with training loss at 0.0367 and validation loss at 0.0353.
The weights in `model.h5` represent these values.