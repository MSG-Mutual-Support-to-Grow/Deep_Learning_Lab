import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical

(x_train,y_train),(x_test,y_test)= tf.keras.datasets.cifar10.load_data()

x_train, x_test =x_train/255.0, x_test/255.0

y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

x_train[0].shape



y_train[0].shape

plt.imshow(x_train[46])

model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    # layers.Conv2D(128,(3,3),activation='relu'),
    # layers.MaxPooling2D((2,2)),
    layers.Conv2D(256,(3,3),activation='relu', input_shape=(32,32,3)),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(10,activation='softmax')
    ])

model.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=["accuracy"]
)

model.summary()

model_train=model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32
)

test_loss,test_acc = model.evaluate(x_test,y_test)
print("Accuracy:",test_acc)
print("Loss:",test_loss)

plt.figure(figsize=(12, 4))

plt.plot(model_train.history['accuracy'], label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')

plt.plot(model_train.history['loss'], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()