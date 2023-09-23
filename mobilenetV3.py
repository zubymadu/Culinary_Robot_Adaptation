import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
import matplotlib.pyplot as plt

# Set GPU memory consumption growth to avoid memory errors and optimize the use of resources
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

root_dir = r'C:\Users\Madu\Desktop\cvprojects\kitchenwares'

# Define path to image directories
train_dir = os.path.join(root_dir, 'train')
val_dir = os.path.join(root_dir, 'test')
# val1_dir = os.path.join(root_dir, 'validation1')
        
# Check number of images in each folder
train_size = sum([len(files) for r, d, files in os.walk(train_dir)])
val_size = sum([len(files) for r, d, files in os.walk(val_dir)])
# test_size = sum([len(files) for r, d, files in os.walk(test_dir)])
print(f'Train size: {train_size}')
print(f'Validation size: {val_size}')
# print(f'Test size: {test_size}')

             
# Define data generators for train, validation, and test sets
train_datagen = ImageDataGenerator(
    rescale=1./255,       # normalize pixel values between 0 and 1
    rotation_range=40,    # randomly rotate images by up to 40 degrees
    width_shift_range=0.2,  # randomly shift image horizontally by up to 20%
    height_shift_range=0.2, # randomly shift image vertically by up to 20%
    shear_range=0.4,         # randomly apply shearing transformations
    zoom_range=0.2,          # randomly zoom in/out on images
    horizontal_flip=True,   # randomly flip images horizontally
    fill_mode='nearest',      # fill any missing pixels with the nearest value
    brightness_range=(0.4, 1.5),   # randomly change brightness of images by up to 40%
    channel_shift_range=30.0,     # randomly shift the color channels
    vertical_flip=True,          # randomly flip images vertically
    
)

valid_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

# Define paths to image directories
train_path = os.path.join(root_dir, 'train')
valid_path = os.path.join(root_dir, 'test')
# test_path = os.path.join(root_dir, 'test')

# Use generators to load images from directories
train_batches = train_datagen.flow_from_directory(
    train_path, 
    target_size=(224, 224), # resize images to (224, 224) during preprocessing
    batch_size=32, 
    
)
# Extract the labels from the generator
train_labels = list(train_batches.class_indices.keys())


valid_batches = valid_datagen.flow_from_directory(
    valid_path,
    target_size=(224, 224),
    batch_size=16,
    
)

# test_batches = test_datagen.flow_from_directory(
#     test_path,
#     target_size=(224, 224),
#     batch_size=6,
#     shuffle=False
    
# )
# Extract the labels from the generator
# test_labels = list(test_batches.class_indices.keys())

# Print the shapes of the datasets
print('Train shape: ', train_batches[0][0].shape)
# print('Test shape: ',  test_batches[0][0].shape)
print('Validation shape: ', valid_batches[0][0].shape)

# Plot some images from the training dataset and save a copy
plt.figure(figsize=(10, 10))
for i in range(8):
    images, labels = next(train_batches)
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(images[0])
    plt.title(f"{train_labels[np.argmax(labels[0])]} ({np.argmax(labels[0])})") # show the class name and index of each image sample
    plt.axis("on")
plt.savefig('sample_images.png')
plt.show()


# Load the pre-trained MobileNetV3 model
base_model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3), pooling='avg', include_top=False)


# Freeze the convolutional layers
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Add a new classifier different from the built-in classifiers of MobileNetV3
last_layer = base_model.output

# Flatten the classifier input from the last layer of the MobileNetV3 model
x = Flatten()(last_layer)

#  Add a fully connected layer of 32 units and batchnorm, dropout and softmax layers
x = Dense(64, activation='relu', name='FC_2')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(6, activation='softmax', name='project')(x)

# Define the new model
new_model = Model(inputs=base_model.input, outputs=x)
new_model.summary()


# Compile the model with categorical cross-entropy loss, Adam optimizer, and learning rate of 1e-4
new_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

# Define callbacks
early_stop = EarlyStopping(patience=5)
model_checkpoint = ModelCheckpoint('best_model.hdf5', verbose=2, save_best_only=True)

# Fit the model with 1000 epochs
hist = new_model.fit(train_batches, epochs=1000, steps_per_epoch=20, validation_data=valid_batches, callbacks=[early_stop, model_checkpoint])

train_acc = new_model.evaluate(train_batches)
# test_acc = new_model.evaluate(test_batches)


# Plot the model accuracy evaluation and save a copy of the plot
plt.plot(hist.history['accuracy'], color='teal')
plt.plot(hist.history['val_accuracy'], color='orange')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('model_accuracy.png')
plt.show()

# Plot the model loss evaluation and save a copy of the plot
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('model_loss.png')
plt.show()


