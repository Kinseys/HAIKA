import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.cluster import OPTICS, KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score,
                             calinski_harabasz_score,
                             davies_bouldin_score)
from sklearn.decomposition import PCA
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam
from PIL import Image
import cv2

# ---------------------- Fix Random Seed ----------------------
SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------- CONFIG ----------------------
IMG_SIZE   = (128, 128)
LATENT_DIM = 64
BATCH_SIZE = 16
EPOCHS     = 8
REG_FACTOR = 1e-5
OUTPUT_DIR = r"D:\fly2\all\mcm_64_3264128_1_all"

for sub in ["activations", "heatmaps", "tsne_plots", "heatmaps_epoch"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

# ------------------ Channel Attention -----------------
class ChannelAttention(Layer):
    def __init__(self, ratio=8):
        super().__init__()
        self.ratio = ratio
    def build(self, input_shape):
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.fc = tf.keras.Sequential([
            layers.Dense(input_shape[-1] // self.ratio, activation='relu'),
            layers.Dense(input_shape[-1], activation='sigmoid')
        ])
        super().build(input_shape)
    def call(self, inputs):
        avg_out = self.fc(self.avg_pool(inputs))
        max_out = self.fc(self.max_pool(inputs))
        return layers.multiply([inputs, avg_out + max_out])

# ------------------ Residual Block ------------------
def residual_block(x, filters):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same',
                                 kernel_regularizer=regularizers.l2(REG_FACTOR))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Conv2D(filters, 3, padding='same',
                      kernel_regularizer=regularizers.l2(REG_FACTOR))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same',
                      kernel_regularizer=regularizers.l2(REG_FACTOR))(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([shortcut, x])
    return layers.Activation('relu')(x)

# -------- Enhanced Autoencoder (with CA) ------------
def build_enhanced_autoencoder(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)

    filters_list = [384, 128, 64]
    for f in filters_list:
        x = residual_block(x, f)
        x = ChannelAttention()(x)
        x = layers.MaxPooling2D(2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    encoded = layers.Dense(LATENT_DIM, activation='relu',
                           kernel_regularizer=regularizers.l2(REG_FACTOR))(x)

    # Decoder
    d = layers.Dense(8 * 8 * 128, activation='relu')(encoded)
    d = layers.Reshape((8, 8, 128))(d)
    for f in reversed(filters_list):
        d = layers.UpSampling2D(2)(d)
        d = residual_block(d, f)
        d = ChannelAttention()(d)
    d = layers.UpSampling2D(2)(d)
    decoded = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(d)

    return Model(inputs, decoded), Model(inputs, encoded)

# ------------------- Data Loading --------------------
def load_data(folder_path):
    images, labels, species, filenames = [], [], [], []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".png"):
            base = os.path.splitext(fname)[0]
            family, species_name = base, ""
            if '-' in base:
                parts = base.split('-')
                family = parts[0].strip()
                species_name = "-".join(parts[1:]).strip() if len(parts) > 1 else ""
            try:
                img = Image.open(os.path.join(folder_path, fname)).convert('L')
                img = img.resize(IMG_SIZE)
                img_array = np.array(img) / 255.0
            except Exception as e:
                print(f"Error loading {fname}: {e}")
                continue
            images.append(img_array[..., np.newaxis])
            labels.append(family)
            species.append(species_name)
            filenames.append(fname)
    return (np.array(images), np.array(labels),
            np.array(species), np.array(filenames))

# ------------- Feature Activation Visualization -------
def visualize_activations(model, img_array, filename):
    layer_outputs = [l.output for l in model.layers if 'conv' in l.name]
    activation_model = Model(model.input, layer_outputs)
    activations = activation_model.predict(img_array[np.newaxis, ...])
    plt.figure(figsize=(20, 5))
    base = os.path.splitext(filename)[0]
    for i, act in enumerate(activations):
        plt.subplot(1, len(activations), i+1)
        plt.imshow(act[0, :, :, 0], cmap='viridis')
        plt.title(f'Layer {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "activations", f"{base}_activations.png"))
    plt.close()

# ------------------ Attention Heatmap ----------------
def generate_heatmap(model, img_array, filename):
    grad_model = Model(model.input,
                       [model.get_layer('channel_attention').output, model.output])
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(img_array[np.newaxis, ...])
        pred = tf.reduce_mean(pred)
    grads = tape.gradient(pred, conv_out)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = sum(w * conv_out[0, :, :, i] for i, w in enumerate(weights))
    cam = cv2.resize(cam.numpy(), IMG_SIZE)
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    plt.figure(figsize=(10, 10))
    plt.imshow(img_array[..., 0], cmap='gray')
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.colorbar(label='Attention')
    base = os.path.splitext(filename)[0]
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmaps", f"{base}_heatmap.png"))
    plt.close()

# ------------------ Epoch Heatmap Generation ----------
def generate_epoch_heatmap(model, img_array, filename):
    grad_model = Model(model.input,
                       [model.get_layer('channel_attention').output, model.output])
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(img_array[np.newaxis, ...])
        pred = tf.reduce_mean(pred)
    grads = tape.gradient(pred, conv_out)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = sum(w * conv_out[0, :, :, i] for i, w in enumerate(weights))
    cam = cv2.resize(cam.numpy(), IMG_SIZE)
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    plt.figure(figsize=(10, 10))
    plt.imshow(img_array[..., 0], cmap='gray')
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.colorbar(label='Attention')
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmaps_epoch", filename))
    plt.close()

class HeatmapEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, sample_img, prefix):
        super().__init__()
        self.sample_img = sample_img
        self.prefix = prefix
    def on_epoch_end(self, epoch, logs=None):
        fname = f"{self.prefix}_epoch_{epoch+1}.png"
        generate_epoch_heatmap(self.model, self.sample_img, fname)

# ------------------- Optimized Clustering -------------
def optimized_clustering(features):
    if len(features) > 10:
        sil = []
        for k in range(2, min(20, len(features)//2)):
            lab = KMeans(k, random_state=SEED).fit_predict(features)
            sil.append(silhouette_score(features, lab))
        best_k = np.argmax(sil)+2
    optics = OPTICS(min_samples=5, metric='cosine')
    return optics.fit_predict(features)

# --------------------- MAIN ---------------------------
def main():
    data_folder = r"D:\fly2\all\11"
    images, families, species, filenames = load_data(data_folder)
    unique_fams = sorted(set(families))

    opt = Adam(learning_rate=1e-3)
    autoencoder, encoder = build_enhanced_autoencoder(images[0].shape)
    autoencoder.compile(opt, loss='mse')

    autoencoder.fit(images, images,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    shuffle=True,
                    validation_split=0.0,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=20),
                        HeatmapEpochCallback(images[0], "sample")
                    ])

    autoencoder.save(os.path.join(OUTPUT_DIR, "autoencoder.h5"))
    encoder.save(os.path.join(OUTPUT_DIR, "encoder.h5"))

    latent = encoder.predict(images)

    # ------------- Save Embeddings TXT + XLSX ----------
    emb_rows = []
    with open(os.path.join(OUTPUT_DIR, "image_embeddings.txt"),
              "w", encoding="utf-8") as f:
        for i, fname in enumerate(filenames):
            img_name = os.path.splitext(fname)[0]          # remove .png
            emb_str  = " ".join(f"{v:.3f}" for v in latent[i])
            f.write(f"{img_name}\t{emb_str}\n")
            emb_rows.append({"Image": img_name, "Embedding": emb_str})
    pd.DataFrame(emb_rows).to_excel(
        os.path.join(OUTPUT_DIR, "image_embeddings.xlsx"), index=False)

    # ------------- Visualizations ----------------------
    for img, fname in zip(images, filenames):
        visualize_activations(encoder, img, fname)
        generate_heatmap(autoencoder, img, fname)

    # ------------- Clustering & t‑SNE ------------------
    clusters_optics = optimized_clustering(latent)
    tsne = TSNE(2, random_state=SEED).fit_transform(latent)

    # t‑SNE scatter coloured by FAMILY  -----------------
    fam_color = {f:i for i,f in enumerate(unique_fams)}
    colours   = [fam_color[f] for f in families]
    plt.figure(figsize=(15, 10))
    sc = plt.scatter(tsne[:,0], tsne[:,1], c=colours,
                     cmap='tab20', alpha=0.8)
    handles = [plt.Line2D([], [], marker='o', linestyle='',
                          markerfacecolor=sc.cmap(fam_color[f]/len(unique_fams)),
                          label=f) for f in unique_fams]
    plt.legend(handles=handles, title="Family",
               bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.title("t‑SNE coloured by Family (token before '-')")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_plots", "tsne_by_family.png"))
    plt.close()

    # t‑SNE with OPTICS
    plt.figure(figsize=(15, 10))
    plt.scatter(tsne[:,0], tsne[:,1], c=clusters_optics,
                cmap='tab20', alpha=0.7)
    plt.colorbar()
    plt.title("t‑SNE with OPTICS")
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_plots", "cluster_visualization_optics.png"))
    plt.close()

    # ---------- KMeans & Agglo (fixed n_clusters) ------
    k30 = KMeans(30, random_state=SEED).fit_predict(latent)
    a30 = AgglomerativeClustering(30).fit_predict(latent)

    for lab, name in [(k30, "kmeans_30"), (a30, "agglo_30")]:
        plt.figure(figsize=(15, 10))
        plt.scatter(tsne[:,0], tsne[:,1], c=lab,
                    cmap='tab20', alpha=0.7)
        plt.colorbar()
        plt.title(f"t‑SNE {name}")
        plt.savefig(os.path.join(OUTPUT_DIR, "tsne_plots",
                                 f"cluster_visualization_{name}.png"))
        plt.close()

    # ----- Build original clustering result table ------
    res_df = pd.DataFrame({
        "Filename"      : filenames,
        "Family"        : families,
        "Species"       : species,
        "Cluster_OPTICS": clusters_optics,
        "tSNE_X"        : tsne[:,0],
        "tSNE_Y"        : tsne[:,1],
        "KMeans_30"     : k30,
        "Agglo_30"      : a30
    })
    res_df.to_excel(os.path.join(OUTPUT_DIR, "clustering_results.xlsx"), index=False)

    # ===================================================
    # === NEW PART : family‑wise clustering (n_fams)   ===
    # ===================================================
    n_fams = len(unique_fams)
    k_fam  = KMeans(n_fams, random_state=SEED).fit_predict(latent)
    a_fam  = AgglomerativeClustering(n_fams).fit_predict(latent)
    fam_res_df = pd.DataFrame({
        "Filename"  : filenames,
        "Family"    : families,
        "KMeans_fam": k_fam,
        "Agglo_fam" : a_fam
    })
    fam_res_df.to_excel(os.path.join(OUTPUT_DIR, "family_clustering_results.xlsx"),
                        index=False)

    # ===================================================
    # === PCA variance for 3 largest families         ===
    # ===================================================
    fam_counts = pd.Series(families).value_counts().head(3).index.tolist()

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    for idx, fam in enumerate(fam_counts):
        fam_idx = np.where(families == fam)[0]
        if len(fam_idx) < 2:
            axes[idx].text(0.5, 0.5, f"Not enough data for {fam}",
                           ha='center', va='center')
            axes[idx].axis('off')
            continue
        pca_f = PCA(n_components=min(10, len(fam_idx), latent.shape[1]),
                    random_state=SEED).fit(latent[fam_idx])
        var_pct = pca_f.explained_variance_ratio_ * 100
        axes[idx].bar(range(1, len(var_pct)+1), var_pct)
        axes[idx].set_title(f"{fam}: variance explained by first "
                            f"{len(var_pct)} PCs")
        axes[idx].set_xlabel("Principal component")
        axes[idx].set_ylabel("Variance %")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_plots",
                             "pca_variance_top3_families.png"))
    plt.close()

    # ===================================================
    # === PCA scatter by family (kept from before)     ===
    # ===================================================
    pca10 = PCA(n_components=10, random_state=SEED)
    pca_res = pca10.fit_transform(latent)

    plt.figure(figsize=(15, 10))
    sc = plt.scatter(pca_res[:,0], pca_res[:,1],
                     c=colours, cmap='tab20', alpha=0.8)
    handles = [plt.Line2D([], [], marker='o', linestyle='',
                          markerfacecolor=sc.cmap(fam_color[f]/len(unique_fams)),
                          label=f) for f in unique_fams]
    plt.legend(handles=handles, title="Family",
               bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.title("PCA coloured by Family (token before '-')")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_plots", "pca_by_family.png"))
    plt.close()

if __name__ == "__main__":
    main()
