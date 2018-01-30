import csv,cv2,random,time,math
from keras.layers import Dense, Flatten, Lambda, Convolution2D, pooling, MaxPooling2D, Cropping2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# LOAD DRIVING DATA
use_my_data = True
    
if use_my_data:
    path_split = '\\'
    folder_path = './my_track_data/'
else:
    path_split = '/'
    folder_path = './provided_track_data/'

log_list = []
with open(folder_path + 'driving_log.csv') as driving_log:
    reader = csv.reader(driving_log)
    for line in reader:
        log_list.append(line)

log_list = log_list[1:]
len(log_list)

# DEFINE AND DISPLAY A TEST IMAGE
test_log = log_list[42]
center_image_path = folder_path + 'IMG/' + test_log[0].split(path_split)[-1]

# CONVERT BGR IMAGES TO RGB
def bgr_to_rgb(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# CROP TOP/BOTTOM OF IMAGES
def crop(img):
    return img[60:135,:,:]

# RESIZE IMAGES TO 64x64
def resize(img):
    return cv2.resize(img,(64, 64), interpolation = cv2.INTER_AREA)

# APPLY GAUSSIAN BLUR
def blur(img):
    # apply blur
    return cv2.GaussianBlur(img, (3,3), 0)

# APPLY RANDOM BRIGHTNESS
def random_brightness(img):
    hsv_img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    brightness = random.uniform(0.35,1.0)
    hsv_img[:,:,2] = hsv_img[:,:,2]*brightness
    return cv2.cvtColor(hsv_img,cv2.COLOR_HSV2RGB)

# FLIP IMAGES
def flip(img):
    return cv2.flip(img,1)

# LOAD IMAGES WITH AUGMENTATION
center_images,other_images = [],[]
center_steering,other_steering = [],[]
time_log = time.time()
for i,log in enumerate(log_list):
    if i % 1000 == 0:
        print('')
        print('lines_processed: {}, lines_left: {}, n_center_images_collected: {}, time_elapsed_since_last_step: {:.4f}'.format(i,len(log_list)-i,len(center_images),time.time()-time_log))
        time_log = time.time()
    if i % 50 == 0:
        print('-',end='')

    center_image_path = folder_path + 'IMG/' + log[0].split(path_split)[-1]
    left_image_path = folder_path + 'IMG/' + log[0].split(path_split)[-1]
    right_image_path = folder_path + 'IMG/' + log[0].split(path_split)[-1]
    
    center_image = blur(bgr_to_rgb(resize(crop(cv2.imread(center_image_path)))))
    left_image = blur(bgr_to_rgb(resize(crop(cv2.imread(left_image_path)))))
    right_image = blur(bgr_to_rgb(resize(crop(cv2.imread(right_image_path)))))
    
    # record input commands
    steering,thrust,brake = float(log[3]),float(log[4]),float(log[5])

    correction = random.uniform(0.20,0.30)

    center_images.append(center_image)
    center_steering.append(steering)
    other_images.extend([left_image,right_image])
    other_steering.extend([steering+correction,steering-correction])

# convert lists to numpy arrays
X_center,y_center = np.array(center_images),np.array(center_steering)
X_other,y_other = np.array(other_images),np.array(other_steering)

print('')
print('lines_processed: {}, lines_left: {}, n_center_images_collected: {}, time_elapsed_since_last_step: {:.4f}'.format(i,len(log_list)-i,len(center_images),time.time()-time_log))
print('X_center shape: {}, y_center shape: {}; X_other shape: {}, y_other shape: {}'.format(X_center.shape,y_center.shape,X_other.shape,y_other.shape))

# STEERING DATA DISTRIBUTIO PRIOT TO BALANCING
n_bins = 30
y_freq, y_bins, ignored = plt.hist(y_center,color='r',bins=n_bins,rwidth=0.8)
plt.title('Steering Data Distribution [BEFORE BALANCING]')
plt.show()

offset = 0.17
y_center_steer_left_index = y_center < -offset # turning left
y_center_steer_center_index = (y_center > -offset) & (y_center < offset) # mostly central
y_center_steer_right_index = (y_center > offset) # turning right

y_other_steer_left_index = y_other < -offset # turning left
y_other_steer_center_index = (y_other > -offset) & (y_other < offset) # mostly central
y_other_steer_right_index = (y_other > offset) # turning right

y_left_count,y_center_count,y_right_count = sum(y_center_steer_left_index),sum(y_center_steer_center_index),sum(y_center_steer_right_index)
print('[BEFORE BALANCING]: left steering: {}, center steering: {}, right steering: {}'.format(y_left_count,y_center_count,y_right_count))
print('[BEFORE BALANCING]: X_data shape: {}, y_data shape: {}, y_mean: {:.4f}, y_std: {:.4f}'.format(X_center.shape,y_center.shape,np.mean(y_center),np.std(y_center)))

# VISUALIZING STEERING ANGLES BEFORE BALANCING
plt.bar(['left','center','right'],[y_left_count,y_center_count,y_right_count],)
plt.title('Steering Orientation [BEFORE BALANCING]')
plt.show()

# BALANCING STEERING DATA
# calculate count difference between left/center data, and right/center data
diff_left = y_center_count - y_left_count
diff_right = y_center_count - y_right_count

# all data is consolidated into these two lists
X_data_list,y_data_list = [],[]

# extend original center image/steering data
X_data_list.extend(X_center)
y_data_list.extend(y_center)

# add left/right image/steering data to balance out the data bias
X_data_add_right,y_data_add_right = shuffle(X_other[y_other_steer_right_index],y_other[y_other_steer_right_index],n_samples=diff_right)
X_data_add_left,y_data_add_left = shuffle(X_other[y_other_steer_left_index],y_other[y_other_steer_left_index],n_samples=diff_left)

# extend additional image/steering data
X_data_list.extend(X_data_add_right)
y_data_list.extend(y_data_add_right)
X_data_list.extend(X_data_add_left)
y_data_list.extend(y_data_add_left)

# convert lists into a numpy array
X_data = np.array(X_data_list)
y_data = np.array(y_data_list)

# save fully processed and balanced X and y data to local disk
np.save('X_data',X_data)
np.save('y_data',y_data)

# VISUALIZING STEERING ANGLES AFTER BALANCING
plt.bar(['left','center','right'],[y_left_count,y_center_count,y_right_count],)
plt.title('Steering Orientation [AFTER BALANCING]')
plt.show()

freq,bins,_ = plt.hist(y_data,color='r',bins=n_bins*2,rwidth=0.8)
plt.title('Steering Data Distribution [AFTER BALANCING]')
plt.show()

# LOAD DATA, SHUFFLE, AND SPLIT FOR TRAINING/VALIDATION
# load X & y data and shuffle
X_data = np.load('X_data.npy')
y_data = np.load('y_data.npy')
X_data,y_data = shuffle(X_data,y_data,n_samples=10000)

# split the data up for training/testing
X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size = 0.20, random_state = 42)

print('[TRAINING AND VALIDATION DATA]: X_train shape: {}, y_train shape: {}, X_valid shape: {}, y_valid shape: {}'.format(X_train.shape,y_train.shape,X_valid.shape,y_valid.shape))
print('[TRAINING AND VALIDATION DATA]: y_train mean: {:.4f}, y_train std: {:.4f}, y_valid mean: {:.4f}, y_valid std: {:.4f}'.format(np.mean(y_train),np.std(y_train),np.mean(y_valid),np.std(y_valid)))

# GENERATORS
# model generators used to minimize memory usage when training model
def train_generator(X,y,batch_size):
    X_batch = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
    y_batch = np.zeros((batch_size,), dtype = np.float32)
    X_shuffled, y_shuffled = shuffle(X, y)
    while True:
        for i in range(batch_size):
            rand = int(np.random.choice(len(X_shuffled),1))
            X_batch[i] = random_brightness(X_shuffled[rand])
            y_batch[i] = y_shuffled[rand]*np.random.uniform(0.90,1.10)

            coin = random.randint(0,1)
            if coin == 1:
                X_batch[i], y_batch[i] = flip(X_batch[i]),-1*y_batch[i]

        yield X_batch, y_batch

# model generators used to minimize memory usage when training model
def valid_generator(X,y,batch_size):
    X_batch = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
    y_batch = np.zeros((batch_size,), dtype = np.float32)
    X_shuffled, y_shuffled = shuffle(X, y)
    while True:
        for i in range(batch_size):
            rand = int(np.random.choice(len(X_shuffled),1))
            X_batch[i] = X_shuffled[rand]
            y_batch[i] = y_shuffled[rand]

        yield X_batch, y_batch


# HYPER-PARAMETERS AND MODEL PIPELINE
# model hyperparameters
BATCH_SIZE = 128
EPOCHS = 3
LR = 0.001

# initialize generators
train_gen = train_generator(X_train, y_train, BATCH_SIZE)
valid_gen = valid_generator(X_valid, y_valid, BATCH_SIZE)

# NVIDIA INSPIRED NETWORK ARCHITECTURE
model = Sequential()
model.add(Lambda(lambda x: (x/255) - 0.5,input_shape=X_train.shape[1:])) # normalization layer
model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), activation='relu', W_regularizer = l2(LR)))
model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2), activation='relu', W_regularizer = l2(LR)))
model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2), activation='relu', W_regularizer = l2(LR)))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2,2), activation='relu', W_regularizer = l2(LR)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(2,2), activation='relu', W_regularizer = l2(LR)))
model.add(Flatten())
model.add(Dense(80, W_regularizer = l2(LR)))
model.add(Dropout(0.5))
model.add(Dense(40, W_regularizer = l2(LR)))
model.add(Dropout(0.5))
model.add(Dense(20, W_regularizer = l2(LR)))
model.add(Dropout(0.5))
model.add(Dense(10, W_regularizer = l2(LR)))
model.add(Dense(1, W_regularizer = l2(LR)))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()
model.fit_generator(train_gen, samples_per_epoch = len(X_train), epochs=EPOCHS, validation_data=valid_gen, nb_val_samples = len(X_valid))

# SAVE TRAINED MODEL
model.save('model.h5')