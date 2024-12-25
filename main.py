import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset.csv')
df = df.drop("dominionVictoryScore", axis=1)
df['teamId'] = df['teamId'].replace({200:1, 100:0})
df['win'] = df['win'].replace({'Win':1, 'Fail':0})
df['firstBlood'] = df['firstBlood'].astype(int)
df['firstTower'] = df['firstTower'].astype(int)
df['firstInhibitor'] = df['firstInhibitor'].astype(int)
df['firstBaron'] = df['firstBaron'].astype(int)
df['firstDragon'] = df['firstDragon'].astype(int)
df['firstRiftHerald'] = df['firstRiftHerald'].astype(int)
scaler = MinMaxScaler()
df['towerKills'] = scaler.fit_transform(df[["towerKills"]])
df['inhibitorKills'] = scaler.fit_transform(df[["inhibitorKills"]])
df['baronKills'] = scaler.fit_transform(df[["baronKills"]])
df['dragonKills'] = scaler.fit_transform(df[["dragonKills"]])
df['riftHeraldKills'] = scaler.fit_transform(df[["riftHeraldKills"]])
df['vilemawKills'] = scaler.fit_transform(df[["vilemawKills"]])
X = df.drop(columns=["win", "vilemawKills"])
y = df["win"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(256, input_dim=X_train.shape[1], activation='relu', name='dense'),
    BatchNormalization(name='batch_normalization'),
    Dropout(0.3, name='dropout'),
    
    Dense(128, activation='relu', name='dense_1'),
    BatchNormalization(name='batch_normalization_1'),
    Dropout(0.3, name='dropout_1'),
    
    Dense(64, activation='relu', name='dense_2'),
    BatchNormalization(name='batch_normalization_2'),
    Dropout(0.3, name='dropout_2'),
    
    Dense(32, activation='relu', name='dense_3'),
    
    Dense(1, activation='sigmoid', name='dense_4')
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=10,
    validation_split=0.2,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Точность модели на тестовых данных: {accuracy}")

model.save('model.keras')