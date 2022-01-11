import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle

def convert_sparse_matrix_to_sparse_tensor(X):
    """Function that converts feature matrix to tensor.

    """
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape))

# Assing data as data and labels
data = pd.read_csv('data.csv')
texts = data['1']
labels = data['0']

# Split data to train and test data
train_data, test_data, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2)

# Define vectorizer and make feature matrixes
vectorizer = CountVectorizer(binary=True, ngram_range=(1, 1))
feature_matrix_train = vectorizer.fit_transform(train_data)
feature_matrix_test = vectorizer.transform(test_data)

feature_matrix_train_tf = convert_sparse_matrix_to_sparse_tensor(
    feature_matrix_train)
feature_matrix_test_tf = convert_sparse_matrix_to_sparse_tensor(
    feature_matrix_test)

# Encode labels
label_encoder = LabelEncoder()
class_numbers_train = label_encoder.fit_transform(train_labels)
class_numbers_test = label_encoder.transform(test_labels)

data_count, feature_count = feature_matrix_train.shape
class_count = len(label_encoder.classes_)

# Define and compile neural network
model = Sequential()
model.add(Dense(3000, input_dim=feature_count, activation='tanh'))
model.add(Dense(3000, activation='tanh'))
model.add(Dense(class_count, activation='softmax'))

model.compile(optimizer="sgd",
              loss="sparse_categorical_crossentropy", metrics=['accuracy'])
hist = model.fit(feature_matrix_train_tf, class_numbers_train, validation_data=(
    feature_matrix_test_tf, class_numbers_test), batch_size=100, epochs=150)

# Plot loss
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range (1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Clear plot
plt.clf()

# Plot accuracy
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
plt.plot(epochs, acc ,'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# serialize to JSON
json_file = model.to_json()
with open("json_model.json", "w") as file:
   file.write(json_file)

# serialize weights to HDF5
model.save_weights("model_weights.h5")