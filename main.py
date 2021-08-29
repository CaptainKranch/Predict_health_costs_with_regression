import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

dataset = pd.read_csv('insurance.csv')

print(dataset.tail())
print(dataset.region.value_counts())

dataset = dataset.join(pd.get_dummies(dataset.region, prefix='region')).drop('region', axis=1)
print(dataset.head())


labels = {}

values = dataset.sex.astype('category')
labels['sex'] = values.cat.categories
dataset['sex'] = values.cat.codes

# print(dataset.head())
# print(labels)

values = dataset.smoker.astype('category')
labels['smoker'] = values.cat.categories
dataset['smoker'] = values.cat.codes

# print(dataset.head())
# print(labels)

var = sns.heatmap(dataset.corr(), annot=True, fmt='.2f')
var = dataset.drop(['region_northeast', 'region_northwest', 'region_southeast', 'region_southwest'], axis=1, inplace=True)
var = dataset.drop(['sex', 'children'], axis=1, inplace=True)
var = sns.pairplot(dataset)

dataset = dataset.sample(frac=1)

size = int(len(dataset) * .2)
train_dataset = dataset[:-size]
test_dataset  = dataset[-size:]

print(len(dataset), len(train_dataset), len(test_dataset))

train_labels  = train_dataset['expenses']
train_dataset = train_dataset.drop('expenses', axis=1)

test_labels   = test_dataset['expenses']
test_dataset  = test_dataset.drop('expenses', axis=1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(len(train_dataset.keys()),)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(
    optimizer= tf.keras.optimizers.RMSprop(0.05),
    loss='mse',
    metrics=['mae', 'mse']
)
model.summary()

class EpochDots(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0:
      print()
      print('Epoch: {:d}, '.format(epoch), end='')
      for name, value in sorted(logs.items()):
        print('{}:{:0.4f}'.format(name, value), end=', ')
      print()

    print('.', end='')


r = model.fit(train_dataset, train_labels, epochs=500,
              verbose=0, callbacks=[EpochDots()])



res = model.evaluate(test_dataset, test_labels, verbose=2)
print(res)

loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)
