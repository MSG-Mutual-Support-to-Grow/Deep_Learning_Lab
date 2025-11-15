import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt

url="/content/House Price Dataset.csv"
data=pd.read_csv(url)

data

x=data[['bedrooms','sqft_living']]
y=data['price']/100000

model = Sequential([
    layers.Dense(64,activation='relu',input_shape=[len(x.keys())]),
    layers.Dense(64,activation='relu'),
    layers.Dense(1)
])

model.compile(loss='mse',optimizer='adam',metrics=['mae'])

history = model.fit(x,y,epochs=10,batch_size=128)

# Get loss + mae directly
loss = history.history['loss']
mae = history.history['mae']

# Plot both curves
plt.plot(loss)
plt.plot(mae)
plt.legend(['loss', 'mae'])
plt.xlabel('Epoch')
plt.grid(True)
plt.show()