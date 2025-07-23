import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

train_data_dir = 'C:/Users/pydiv/OneDrive/Desktop/Emotion analysis/Data/train'
validation_data_dir = 'C:/Users/pydiv/OneDrive/Desktop/emotion analysis/Data/test'

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Counting images
num_train_imgs = sum(len(files) for _, _, files in os.walk(train_data_dir))
num_test_imgs = sum(len(files) for _, _, files in os.walk(validation_data_dir))

print("Number of training images:", num_train_imgs)
print("Number of validation images:", num_test_imgs)

# Set the batch size
batch_size = 32
if num_train_imgs < batch_size:
    batch_size = num_train_imgs

steps_per_epoch = num_train_imgs // batch_size
validation_steps = num_test_imgs // batch_size

print("Steps per epoch:", steps_per_epoch)
print("Validation steps:", validation_steps)

# Building the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Fit the model
history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=100,
                    validation_data=validation_generator,
                    validation_steps=validation_steps)

model.save('model_file1.h5')
  