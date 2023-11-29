import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import seaborn as sns
import csv
import tensorflow as tf
from tensorflow import keras

# Load data using pandas
df = pd.read_csv("new_merged.csv")

# Extract relevant columns
solar_mass = df['Solar Mass'].str.replace("''", '').str.replace(',', '').str.replace('–', '-').apply(lambda x: eval(x) if pd.notna(x) else np.nan)
solar_radius = df['Solar Radius'].str.replace("''", '').str.replace(',', '').str.replace('–', '-').apply(lambda x: eval(x) if pd.notna(x) else np.nan)

# Filter out missing values
filtered_data = df[['Solar Mass', 'Solar Radius']].dropna(subset=['Solar Mass', 'Solar Radius'])

# Convert data to numeric
filtered_data['Solar Mass'] = pd.to_numeric(filtered_data['Solar Mass'], errors='coerce')
filtered_data['Solar Radius'] = pd.to_numeric(filtered_data['Solar Radius'], errors='coerce')

# Combine mass and radius into a feature matrix X
X = filtered_data[['Solar Radius', 'Solar Mass']].values

# Perform k-means clustering using scikit-learn
wcss_sklearn = []
for i in range(1, 11):
    kmeans_sklearn = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans_sklearn.fit(X)
    wcss_sklearn.append(kmeans_sklearn.inertia_)

# Perform k-means clustering using TensorFlow and Keras
wcss_tensorflow = []
for i in range(1, 11):
    kmeans_tensorflow = keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(i, activation='softmax')
    ])
    kmeans_tensorflow.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    kmeans_tensorflow.fit(X, np.zeros(len(X)), epochs=10, verbose=0)
    wcss_tensorflow.append(kmeans_tensorflow.get_layer(index=0).get_weights()[0].shape[0])

# Plot the elbow method
plt.figure(figsize=(15, 5))

# Plot scikit-learn results
plt.subplot(1, 2, 1)
sns.lineplot(range(1, 11), wcss_sklearn, marker='o', color='red')
plt.title('The elbow method (scikit-learn)')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

# Plot TensorFlow and Keras results
plt.subplot(1, 2, 2)
sns.lineplot(range(1, 11), wcss_tensorflow, marker='o', color='blue')
plt.title('The elbow method (TensorFlow and Keras)')
plt.xlabel('Number of clusters')
plt.ylabel('Number of neurons in the dense layer')

plt.tight_layout()
plt.show()
