import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.cluster import OPTICS, KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA

from tensorflow.keras import layers, Model, regularizers
from skimage.draw import ellipsoid

# ---------------- CONFIG ----------------
SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

LATENT_DIM = 32
BATCH_SIZE = 8
EPOCHS = 40
REG_FACTOR = 1e-5
OUTPUT_DIR = r"D:\fly3d\synthetic_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "tsne_plots"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)

LABEL_TEXTS = {
    0: "Sphere",
    1: "Cube",
    2: "Ellipsoid",
    3: "Cylinder",
    4: "Torus",
}

# -------------- Synthetic 3D Shape Generator --------------

def make_shape(shape_type, size=28):
    vol = np.zeros((size, size, size), dtype=np.float32)
    center = np.array(vol.shape) / 2

    if shape_type == "Sphere":
        rr, cc, zz = np.indices(vol.shape)
        mask = (rr - center[0])**2 + (cc - center[1])**2 + (zz - center[2])**2 < (size/4)**2
        vol[mask] = 1

    elif shape_type == "Cube":
        s = size // 4
        start = size // 2 - s
        end = size // 2 + s
        vol[start:end, start:end, start:end] = 1

    elif shape_type == "Ellipsoid":
        e = ellipsoid(size//4, size//6, size//5, levelset=True).astype(np.float32)
        pad_width = [( (size - e.shape[i])//2, (size - e.shape[i] + 1)//2 ) for i in range(3)]
        vol = np.pad(e, pad_width, mode='constant')

    elif shape_type == "Cylinder":
        rr, cc, zz = np.indices(vol.shape)
        mask = (rr - center[0])**2 + (cc - center[1])**2 < (size/4)**2
        height_mask = (zz > size//4) & (zz < 3*size//4)
        vol[mask & height_mask] = 1

    elif shape_type == "Torus":
        rr, cc, zz = np.indices(vol.shape)
        R = size/4   # major radius
        r = size/8   # minor radius
        x = rr - center[0]
        y = cc - center[1]
        z = zz - center[2]
        mask = (np.sqrt(x**2 + y**2) - R)**2 + z**2 < r**2
        vol[mask] = 1

    return vol[..., np.newaxis]

def make_synthetic_3d_dataset(n_samples_per_class=200, size=28):
    images = []
    labels = []
    for label, name in LABEL_TEXTS.items():
        for _ in range(n_samples_per_class):
            vol = make_shape(name, size=size)
            # Add small random noise
            vol = vol + np.random.normal(0, 0.05, vol.shape)
            vol = np.clip(vol, 0, 1)
            images.append(vol)
            labels.append(label)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)
    return images, labels

# -------------- Model ------------------

class ChannelAttention3D(layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        C = int(input_shape[-1])
        self.avg_pool = layers.GlobalAveragePooling3D()
        self.max_pool = layers.GlobalMaxPooling3D()
        self.fc1 = layers.Dense(max(C // self.ratio, 1), activation='relu')
        self.fc2 = layers.Dense(C, activation='sigmoid')
        super().build(input_shape)

    def call(self, inputs):
        avg = self.fc2(self.fc1(self.avg_pool(inputs)))
        mx = self.fc2(self.fc1(self.max_pool(inputs)))
        att = tf.expand_dims(tf.expand_dims(tf.expand_dims(avg + mx, 1), 1), 1)
        return inputs * att

def residual_block_3d(x, filters):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv3D(filters, (1,1,1), padding='same',
                                 kernel_regularizer=regularizers.l2(REG_FACTOR))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.Conv3D(filters, (3,3,3), padding='same',
                      kernel_regularizer=regularizers.l2(REG_FACTOR))(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv3D(filters, (3,3,3), padding='same',
                      kernel_regularizer=regularizers.l2(REG_FACTOR))(y)
    y = layers.BatchNormalization()(y)
    y = layers.add([shortcut, y])
    y = layers.Activation('relu')(y)
    return y

def build_autoencoder_3d(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv3D(32, (3,3,3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D((2,2,2))(x) # 28->14

    filters_list = [64, 96]
    for f in filters_list:
        x = residual_block_3d(x, f)
        x = ChannelAttention3D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.MaxPooling3D((2,2,2))(x) # 14->7

    x = layers.GlobalAveragePooling3D()(x)
    encoded = layers.Dense(LATENT_DIM, activation='relu',
                           kernel_regularizer=regularizers.l2(REG_FACTOR))(x)

    d = layers.Dense(7*7*7*128, activation='relu')(encoded)
    d = layers.Reshape((7,7,7,128))(d)

    for f in reversed(filters_list):
        d = layers.UpSampling3D((2,2,2))(d)
        d = residual_block_3d(d, f)
        d = ChannelAttention3D()(d)

    decoded = layers.Conv3D(1, (3,3,3), activation='sigmoid', padding='same')(d)

    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    return autoencoder, encoder

# -------------- Clustering Utils --------------

def optimized_clustering(features):
    optics = OPTICS(min_samples=5, metric='cosine')
    return optics.fit_predict(features)

def compute_cluster_metrics(features, labels):
    if len(set(labels)) < 2:
        return {"Silhouette": None, "Calinski_Harabasz": None, "Davies_Bouldin": None}
    return {
        "Silhouette": silhouette_score(features, labels),
        "Calinski_Harabasz": calinski_harabasz_score(features, labels),
        "Davies_Bouldin": davies_bouldin_score(features, labels),
    }

# -------------- MAIN ------------------

def main():
    images, labels = make_synthetic_3d_dataset(n_samples_per_class=200)
    filenames = np.array([f"sample_{i:05d}.npz" for i in range(len(images))])

    autoencoder, encoder = build_autoencoder_3d(images[0].shape)
    autoencoder.compile(optimizer='adam', loss='mse')
    es = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    autoencoder.fit(images, images,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    shuffle=True,
                    validation_split=0.1,
                    callbacks=[es])

    autoencoder.save(os.path.join(OUTPUT_DIR, "models", "autoencoder3d.keras"))
    encoder.save(os.path.join(OUTPUT_DIR, "models", "encoder3d.keras"))

    latent_features = encoder.predict(images, batch_size=BATCH_SIZE)

    with open(os.path.join(OUTPUT_DIR, "image_embeddings.txt"), "w") as f:
        for i, fname in enumerate(filenames):
            f.write(f"{LABEL_TEXTS[labels[i]]}\t" +
                    " ".join(map(str, latent_features[i])) + "\n")

    clusters_optics = optimized_clustering(latent_features)
    kmeans_5 = KMeans(n_clusters=5, random_state=SEED).fit_predict(latent_features)
    kmeans_30 = KMeans(n_clusters=30, random_state=SEED).fit_predict(latent_features)
    agglo_30 = AgglomerativeClustering(n_clusters=30).fit_predict(latent_features)

    tsne = TSNE(n_components=2, random_state=SEED)
    tsne_results = tsne.fit_transform(latent_features)

    # True label plot
    plt.figure(figsize=(15,10))
    for i, txt in enumerate(labels):
        plt.scatter(tsne_results[i,0], tsne_results[i,1], c=f"C{txt}", label=LABEL_TEXTS[txt] if i==np.where(labels==txt)[0][0] else "")
    plt.legend()
    plt.title("t-SNE colored by TRUE labels")
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_plots", "tsne_true_labels.png"))
    plt.close()

    # PCA with true labels
    pca = PCA(n_components=2, random_state=SEED)
    pca_results = pca.fit_transform(latent_features)
    plt.figure(figsize=(15,10))
    for i, txt in enumerate(labels):
        plt.scatter(pca_results[i,0], pca_results[i,1], c=f"C{txt}", label=LABEL_TEXTS[txt] if i==np.where(labels==txt)[0][0] else "")
    plt.legend()
    plt.title("PCA colored by TRUE labels")
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_plots", "pca_true_labels.png"))
    plt.close()

    result_df = pd.DataFrame({
        'Filename': filenames,
        'True_Label': labels,
        'True_Label_Name': [LABEL_TEXTS[l] for l in labels],
        'Cluster_OPTICS': clusters_optics,
        'KMeans_5': kmeans_5,
        'KMeans_30': kmeans_30,
        'Agglo_30': agglo_30,
        'tSNE_X': tsne_results[:,0],
        'tSNE_Y': tsne_results[:,1],
        'PCA_X': pca_results[:,0],
        'PCA_Y': pca_results[:,1],
    })
    result_df.to_excel(os.path.join(OUTPUT_DIR, "clustering_results.xlsx"), index=False)

    metrics_optics = compute_cluster_metrics(latent_features, clusters_optics)
    metrics_k5 = compute_cluster_metrics(latent_features, kmeans_5)
    metrics_k30 = compute_cluster_metrics(latent_features, kmeans_30)
    metrics_a30 = compute_cluster_metrics(latent_features, agglo_30)

    metrics_table = pd.DataFrame({
        "Clustering Method": ["OPTICS", "KMeans_5", "KMeans_30", "Agglo_30"],
        "Silhouette": [metrics_optics["Silhouette"], metrics_k5["Silhouette"], metrics_k30["Silhouette"], metrics_a30["Silhouette"]],
        "Calinski_Harabasz": [metrics_optics["Calinski_Harabasz"], metrics_k5["Calinski_Harabasz"], metrics_k30["Calinski_Harabasz"], metrics_a30["Calinski_Harabasz"]],
        "Davies_Bouldin": [metrics_optics["Davies_Bouldin"], metrics_k5["Davies_Bouldin"], metrics_k30["Davies_Bouldin"], metrics_a30["Davies_Bouldin"]],
        "ARI": [
            adjusted_rand_score(labels, clusters_optics),
            adjusted_rand_score(labels, kmeans_5),
            adjusted_rand_score(labels, kmeans_30),
            adjusted_rand_score(labels, agglo_30)
        ],
        "NMI": [
            normalized_mutual_info_score(labels, clusters_optics),
            normalized_mutual_info_score(labels, kmeans_5),
            normalized_mutual_info_score(labels, kmeans_30),
            normalized_mutual_info_score(labels, agglo_30)
        ]
    })

    metrics_table.to_excel(os.path.join(OUTPUT_DIR, "clustering_metrics.xlsx"), index=False)
    with open(os.path.join(OUTPUT_DIR, "clustering_metrics.txt"), "w") as f:
        f.write(metrics_table.to_string(index=False))

if __name__ == "__main__":
    main()
