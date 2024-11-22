####################################################
#
# MobileNetV2
#
####################################################
print('Model = MobileNetV2')

s = open("Output_MobileNetV2.txt", "w")
s.write('>> Importing Packages...\n')

# Importing all required libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras import Model, layers, models
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, auc, roc_curve
import seaborn as sns
s.write('\n\t\t\t\t....DONE!\n')


# Epochs
ep = 100

# Batch size
bsize_train = 860
bsize_val_test = 110


s.write('>> Preprocessing data...\n')
# Define train, test & validation directory
train_dir = 'Datasets/train'
test_dir = 'Datasets/test'
validation_dir = 'Datasets/validation'


# Use ImageDataGenerator to create variations
train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.1,
                                   horizontal_flip=True, vertical_flip=True, rescale=1/255)

validation_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
                                        height_shift_range=0.2, shear_range=0.2, zoom_range=0.1, horizontal_flip=True, vertical_flip=True, rescale=1/255)

test_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
                                  height_shift_range=0.2, shear_range=0.2, zoom_range=0.1,
                                  horizontal_flip=True, vertical_flip=True, rescale=1/255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(200, 200),
                                                    batch_size=bsize_train,
                                                    class_mode='categorical', shuffle=True)

validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(200, 200), 
                                                              batch_size=bsize_val_test, class_mode='categorical', shuffle=True)

test_generator = test_datagen.flow_from_directory(test_dir, target_size=(200,200),
                                                  batch_size=bsize_val_test,
                                                  class_mode='categorical', shuffle=False)

s.write('\n\t\t\t\t....DONE!\n')


s.write('>> Creating the transfer learning model...\n')
# The transfer learning model
MobileNet = MobileNetV2(input_shape=(200, 200, 3), include_top = False, weights = 'imagenet')
MobileNet.summary()

# Make layers not trainable
for layer in MobileNet.layers[:]:
    layer.trainable = False

# Check layers
for layer in MobileNet.layers:
    print(layer, layer.trainable)

# Remove layers
MobileNet_2 = Model(MobileNet.input, MobileNet.layers[-2].output)
MobileNet_2.summary()


# Make CNN Model
model = models.Sequential()
model.add(MobileNet_2)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(1280, (3, 3), activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(1280, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(2, activation='softmax'))
model.summary()
s.write('\n\t\t\t\t....DONE!\n')


# Optimize / Train & Save
s.write('>> Optimize / Train & Save Model...\n')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(train_generator, steps_per_epoch=100,
                  epochs=ep, validation_data=validation_generator, validation_steps=50)

model.save('MobileNetV2.h5')

s.write('\n\t\t\t\t....DONE!\n')


# Plot result
s.write('>> Generate plots...\n')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('Output_MobileNetV2_Training_and_validation_accuracy.png')
plt.clf()

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('Output_MobileNetV2_Training_and_validation_loss.png')
plt.clf()

s.write('\n\t\t\t\t....DONE!\n')


# Plot confusion matrices
s.write('\n>> Plotting the confusion matrix... \n')

# Print the Target names
target_names = []
for key in train_generator.class_indices:
    target_names.append(key)


# Test Confusion Matrix
Y_pred_test = model.predict(test_generator)
y_pred_test = np.argmax(Y_pred_test, axis=1)
y_test = test_generator.classes
cm_test = confusion_matrix(y_test, y_pred_test)

# Plot & save test confusion matrix
sns.heatmap(cm_test,
            cmap='gnuplot',
            fmt='d',
            annot = True,
            cbar=True,
            xticklabels = target_names,
            yticklabels = target_names)
plt.ylabel('True label', fontsize = 12)
plt.xlabel('Predicted label', fontsize = 12)
plt.title('Test confusion matrix', fontsize = 14)
plt.savefig('Output_MobileNetV2_Test_Confusion_Matrix.png')
plt.clf()

# Print confusion matrix data
print(target_names)
print(cm_test)

s.write('\n\t\t\t\t....DONE!\n')


# Accuracy, Precision, Recall, and F1-scores
s.write('\n>> Print Accuracy, Precision, Recall, and F1-scores... \n')

s.write('\n> Test Accuracy: %0.4f' %(accuracy_score(y_test, y_pred_test)) + '\n')
s.write('\n> Test Precision: %0.4f' %(precision_score(y_test, y_pred_test)) + '\n')
s.write('\n> Test Recall: %0.4f' %(recall_score(y_test, y_pred_test)) + '\n')
s.write('\n> Test F1-scores: %0.4f' %(f1_score(y_test, y_pred_test)) + '\n')

# AUC ROC
y = np.array(y_test)
scores = np.array(y_pred_test)
fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1)
roc_auc = auc(fpr, tpr)
s.write('\n> Test AUC ROC scores: %0.4f' %(roc_auc) + '\n')

s.write('\n\t\t\t\t....DONE!\n')

# Plot AUC ROC curve
s.write('\n>> Plotting AUC ROC curve... \n')
plt.figure(figsize=(8, 8))
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' %(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 12)
plt.ylabel('True Positive Rate', fontsize = 12)
plt.title('Receiver operating characteristic (ROC)', fontsize = 14)
plt.legend(loc="lower right")
plt.savefig('Output_MobileNetV2_Test_AUC_ROC.png')
plt.clf()

s.write('\n\t\t\t\t....DONE!\n')

s.write('\n******************PROGRAM is DONE *******************')
s.close()