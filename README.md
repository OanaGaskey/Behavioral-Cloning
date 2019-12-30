# Behavioral Cloning

Deep Neural Network modeled in Keras to clone human behavior of steering a car around a virtual track using camera images.

![GIF](examples/BehavioralCloning.gif)

If you would teach someone how to drive, how would you go about it?
Would you tell them "if the car is not centered on the road, figure out the offset in meters and then compute the steering angle in radians to compensate the offset knowing that the car's yaw rate is the velocity divided by the distance between the two axels times the tangent of the steering angle"?
or would you say "just steer the car while driving around until you get the *feel* for it"?

This philosophy is at the core of my project. The idea was developed by the researchers at NVIDIA and is explained in this [article](​https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

The end-to-end technique of steering a car is based on convolutional neural networks to map the raw pixels from front-facing cameras to the steering commands for a self-driving car.

I used Udacity's [simulator](https://github.com/udacity/self-driving-car-sim/releases) to manually drive the car around the lake track and record data from the front-facing cameras together with my own steering comands as labels. The data is used to train a convolutional neural network (CNN) that is further used to predict the necessary steering angle and drive the car autonomously around the same track.

This project is implemented in Python using Keras and OpenCV. The source code is located in `model.py` file above. 

To run the code, start the simulator and use `python drive.py model.h5` from the command line.

The starting code for this project is provided by Udacity and can be found [here](https://github.com/udacity/CarND-Behavioral-Cloning-P3)



## Data Aquisition

The simulator setup is the equivalent to the NVIDIA platform pictured below. Three cameras are used, one in the center and one on each side of the windshield. All three cameras are pointed towards the road ahead. 

![sys_arch](examples/sys_arch.JPG)


Data is recorded while driving manually around the track. One recording will result in an `IMG` folder with pictures from all three cameras and a `driving_log.csv` file. In this repository I included data from driving a two laps, each in opposite direction: [fulllap](fulllap); [fulllapcounter](fulllapcounter).

![l_c_r](examples/l_c_r.JPG)


The `driving_log.csv` has the following structure, providing images path and names and recorded comands. Only the `steering angle` is used from the recorded comands.

![data_rec](examples/data_rec.JPG)


Given that multiple recording sessions are needed to build the data set, each path is used to load the data from the `csv` file and build a list with images names and labels.

```
csv_path = path + 'driving_log.csv'
img_folder_path = path + 'IMG/'
with open(csv_path) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        #compose the relative location of images with respect of the current location
        relative_line = line
        #take the last part of the image line separated by '/' this is the image name
        center_img_name = line[0].split('/')[-1]
        #compose the image path by concatenating the folder path and image name
        center_img_path = img_folder_path + center_img_name
        #override the location of the center image
        relative_line[0] = center_img_path
```

For each recorded cycle, 6 images and labels are generated. The central image is initially used in its original form with the corresponding label from the recorded steering angle. The central image is then flipped around its vertical axis and the label's sign is reversed.

```
#CENTER
center_img = cv2.cvtColor(cv2.imread(batch_line[0]), cv2.COLOR_BGR2RGB)
images.append(center_img)
center_label = float(batch_line[3])
labels.append(center_label)
#flip image
flipped_img = cv2.flip(center_img, flipCode=1)
images.append(flipped_img)
#change label sign for flipped image 
flipped_label = (-1.0)* center_label
labels.append(flipped_label)
```

This data augmentation technique has proven to be very efficient. With one recorded image, the model learns to steer both left and right.
Not only that this approach doubles the data set, it helps in compensating for the left turning bias. The lake track in the simulation is a loop, mostly composed of left turns. Because of this, models trained without flipped images result in left steering even on straight road segments. 

 
Images from right and left cameras are also used in their original form, then flipped. Given that the left and right images are taken from different points of view, a correction factor is needed. 

```
#LEFT
correction  = 0.2
left_img = cv2.cvtColor(cv2.imread(batch_line[1]), cv2.COLOR_BGR2RGB)
images.append(left_img)
left_label  = center_label + correction
labels.append(left_label)
#flip image
flipped_img = cv2.flip(left_img, flipCode=1)
images.append(flipped_img)
#change label sign for flipped image 
flipped_label = (-1.0)* left_label
labels.append(flipped_label)
```

When running the model to predict steering angles, only the central camera is used. The left camera provides a shifted perspective. If the left image was seen from the central camera, the labeled steering angle needs to be adjusted to steer towards the right hand side of the road. The correction factor of `0.2` was chosen through trial and error.

This data augmentation technique is used to train the model for recovery situations in which the vehicle drifts to the side. The full laps provided with this repository were driven manually while generally keeping the car in the center of the road. Without the addition of left and right camera pictures, the model would not learn to steer the car from te side, back to the middle of the road. One could think that if the model was trained to drive on the middle of the road that the car will never end up off center. Neverthe less this happends due to imperfect predicted angles and recovery manouvers are essential for this neural network.
 
Even if data augmentation had a big impact it was still not enough to drive the vehicle all the way around the track. I encountered difficulties in sharp curves and I recorded those sections of the track multiple times. I did not include those recordings in this repository due to space limitations. 


## Generators

## Model Architecture

## Model Training

## Simulation Video

[![HighwayDriving](https://img.youtube.com/vi/QEajKfN8Oxo/0.jpg)](https://www.youtube.com/watch?v=QEajKfN8Oxo)
 Click on the image to see the video!