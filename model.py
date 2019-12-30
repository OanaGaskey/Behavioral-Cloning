import numpy as np
import math
import csv
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Conv2D, Activation, Dropout
from keras.callbacks import ModelCheckpoint

def data_loader (path) :
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
            #similar approach for left image
            left_img_name = line[1].split('/')[-1]
            left_img_path = img_folder_path + left_img_name
            relative_line[1] = left_img_path
            #similar approach for right image
            right_img_name = line[2].split('/')[-1]
            right_img_path = img_folder_path + right_img_name
            relative_line[2] = right_img_path
            lines.append(relative_line)

def data_generator (data_lines, batch_size = 128) :
    num_lines = len(data_lines)
    while True:
        shuffle(data_lines)
        for offset in range(0, num_lines, batch_size):
            batch_lines = data_lines[offset:offset + batch_size]
            images = []
            labels = []
            for batch_line in batch_lines:
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
                #RIGHT
                right_img = cv2.cvtColor(cv2.imread(batch_line[2]), cv2.COLOR_BGR2RGB)
                images.append(right_img)
                right_label = center_label - correction
                labels.append(right_label)
                #flip image
                flipped_img = cv2.flip(right_img, flipCode=1)
                images.append(flipped_img)
                #change label sign for flipped image 
                flipped_label = (-1.0)* right_label
                labels.append(flipped_label)

            X_train = np.array(images)
            y_train = np.array(labels)
            yield shuffle(X_train, y_train)
    
lines = []    
#use Udacity provided data
path = '../../../opt/carnd_p3/data/'
data_loader(path)
#use recorded data            
path = 'fulllap/'
data_loader(path)
path = 'fulllapcounter/'
data_loader(path)
path = 'recoveryright1/'
data_loader(path)
path = 'recoveryright2/'
data_loader(path)
path = 'curves/'
data_loader(path)
path = 'curves2/'
data_loader(path)
path = 'curves3/'
data_loader(path)
data_loader(path)
data_loader(path)
            
#print(lines[0])
#print(lines[len(lines)-1])
print('len lines= ',len(lines))

train_data, val_data = train_test_split(lines, test_size = 0.2)
   
model = Sequential()
#add cropping layer
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
#normalize data
model.add(Lambda (lambda x: (x / 255.0) - 0.5) )
#layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="elu"))
#layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2
model.add(Conv2D(36, (5,5), strides=(2, 2), activation="elu"))
#layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
model.add(Conv2D(48, (5,5), strides=(2, 2), activation="elu"))
#layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Conv2D(64, (3,3), activation="elu"))
#layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Conv2D(64, (3,3), activation="elu"))
#flatten image from 2D to side by side
model.add(Flatten())
#layer 6- fully connected layer 1
model.add(Dense(100, activation="elu"))
#dropout layer to avoid overfitting
model.add(Dropout(0.25))
#layer 7- fully connected layer 1
model.add(Dense(50, activation="elu"))
#layer 8- fully connected layer 1
model.add(Dense(10, activation="elu"))
#layer 9- fully connected layer 1
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit_generator(generator = data_generator(train_data),
                    validation_data = data_generator(val_data),
                    epochs = 2,
                    steps_per_epoch  = math.ceil(len(train_data) / 128),
                    validation_steps = math.ceil(len(val_data)   / 128)    )
model.save('model.h5')
print('Model saved')