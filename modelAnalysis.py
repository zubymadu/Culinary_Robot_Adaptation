import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load your pre-trained CNN model
model = tf.keras.models.load_model('best_model.hdf5')  # Replace with your model file

# Define the root directory and paths to image directories
root_dir = r'C:\Users\Madu\Desktop\cvprojects\kitchenwares'
train_dir = os.path.join(root_dir, 'train')
test_dir = os.path.join(root_dir, 'test')  # Updated to 'test' for the test dataset

# Check the number of images in each folder
train_size = sum([len(files) for r, d, files in os.walk(train_dir)])
test_size = sum([len(files) for r, d, files in os.walk(test_dir)])  # Updated for the test dataset
print(f'Train size: {train_size}')
print(f'Test size: {test_size}')  # Updated for the test dataset

# Use data generators to load and preprocess the test dataset
test_datagen = ImageDataGenerator(rescale=1./255)  # You can add more preprocessing if needed

# Load and preprocess the test dataset
test_batches = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),  # Adjust the target size as needed
    batch_size=32,  # Adjust batch size as needed
    shuffle=False  # Important: don't shuffle for evaluation
)

# Make predictions on the test dataset
y_pred = model.predict(test_batches)

# Convert the one-hot encoded labels to class labels
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = test_batches.classes

# Evaluate the model and print a classification report
class_labels = list(test_batches.class_indices.keys())
print(classification_report(y_true_classes, y_pred_classes, target_names=class_labels))

# Generate and display a confusion matrix
confusion_mtx = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)


# Annotate the confusion matrix with numbers
for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j, i, str(confusion_mtx[i, j]), horizontalalignment='center', verticalalignment='center')

plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=45)
plt.yticks(tick_marks, class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# plt.title('Confusion Matrix')
# plt.colorbar()
# tick_marks = np.arange(len(class_labels))
# plt.xticks(tick_marks, class_labels, rotation=45)
# plt.yticks(tick_marks, class_labels)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()

# # You can add more evaluation and visualization as needed
