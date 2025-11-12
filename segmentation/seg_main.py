import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image as PILImage
from tqdm import tqdm
import torch
import kagglehub
import cv2

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow_hub as hub

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ---------------------------- Configuration ---------------------------- #
input_dir = Path(r"D:\fly2\all\11")  # Directory with images
output_dir = Path(r"D:\fly\SEG_all_new")
output_dir.mkdir(parents=True, exist_ok=True)

NUM_CLUSTERS = 20
PCA_COMPONENTS = 10
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

##########################################################
# Step 1: Download and Load SAM model using kagglehub
##########################################################
model_path = kagglehub.model_download("metaresearch/segment-anything/pyTorch/vit-b")
print("Path to model files:", model_path)

checkpoint_path = os.path.join(model_path, "model.pth")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError("model.pth not found in the downloaded model directory.")

device = "cpu"
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
sam.to(device=device)

# Adjust parameters to get more masks if needed
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=None,
    point_grids=[np.array([[0.5, 0.5]])],  # Increase to get more candidate masks
    pred_iou_thresh=0.8,  # Lower this slightly to include more masks
    stability_score_thresh=0.9  # Lower this slightly to include more masks
)

##########################################################
# Load a Pretrained Feature Extractor (MobileNetV2)
##########################################################
# We will use MobileNetV2 from TF Hub to extract image embeddings.
feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
feature_extractor = hub.KerasLayer(feature_extractor_url, trainable=False)


# Ensure model is built in a TF2 eager context
# We'll define a small function to get embeddings from a cropped image region.
def get_embedding_from_image(cropped_img):
    # cropped_img: [H,W,3] uint8
    # Preprocess: resize to 224x224, scale to [0,1]
    img = tf.image.resize(cropped_img, (224, 224))
    img = img / 255.0
    img = tf.expand_dims(img, 0)  # [1,H,W,3]
    emb = feature_extractor(img)  # [1, 1280]
    return emb.numpy().squeeze(0)  # [1280]


##########################################################
# Helper Functions
##########################################################
def parse_species_from_name(image_name):
    return image_name.split('-')[0]


def load_images(image_dir):
    image_paths = [f for f in image_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

    images = []
    image_names = []
    species_labels = []
    for img_path in tqdm(image_paths, desc="Loading Images"):
        try:
            img = PILImage.open(img_path).convert('RGB')
            img_array = np.array(img)
            images.append(img_array)
            image_names.append(img_path.name)
            sp = parse_species_from_name(img_path.name)
            species_labels.append(sp)
        except Exception as e:
            print(f"Error loading {img_path.name}: {e}")
    return images, image_names, species_labels


def plot_dendrogram(linkage_matrix, output_path, labels):
    plt.figure(figsize=(20, max(10, len(labels) * 0.3)))
    dendrogram(linkage_matrix,
               labels=labels,
               leaf_rotation=0.,
               leaf_font_size=10.,
               orientation='left',
               show_contracted=False)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Distance')
    plt.ylabel('Images')
    plt.subplots_adjust(left=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pca_variance(pca, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.title('Cumulative Variance Explained by PCA Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_embeddings_2d(features_2d, labels, title, output_path, species_labels=None):
    plt.figure(figsize=(12, 8))
    if species_labels is not None:
        unique_species = np.unique(species_labels)
        palette = sns.color_palette('tab20', len(unique_species))
        species_to_color = {sp: col for sp, col in zip(unique_species, palette)}
        for i, sp in enumerate(species_labels):
            plt.scatter(features_2d[i, 0], features_2d[i, 1], color=species_to_color[sp], s=20)
        handles = [plt.Line2D([0], [0], marker='o', color=species_to_color[sp], label=sp, linestyle='') for sp in
                   unique_species]
        plt.legend(handles=handles, title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(features_2d[:, 0], features_2d[:, 1], s=20)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_embeddings_3d(features_3d, labels, title, output_path, species_labels=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    if species_labels is not None:
        unique_species = np.unique(species_labels)
        palette = sns.color_palette('tab20', len(unique_species))
        species_to_color = {sp: col for sp, col in zip(unique_species, palette)}
        for i, sp in enumerate(species_labels):
            ax.scatter(features_3d[i, 0], features_3d[i, 1], features_3d[i, 2],
                       color=species_to_color[sp], s=20)
        handles = [plt.Line2D([0], [0], marker='o', color=species_to_color[sp],
                              label=sp, linestyle='') for sp in unique_species]
        ax.legend(handles=handles, title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2], s=20)

    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_cluster_distribution(cluster_labels, output_path):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=cluster_labels, palette='viridis')
    plt.title('Number of Images per Cluster')
    plt.xlabel('Cluster Label')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


##########################################################
# Main Execution
##########################################################
def main():
    images, image_names, species_labels = load_images(input_dir)
    print(f"Total images loaded: {len(images)}")

    mask_info_records = []
    image_embeddings = []

    for i, img in enumerate(tqdm(images, desc="Generating masks and embeddings")):
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        image_name = image_names[i]
        species = species_labels[i]

        # Generate masks for the entire image
        masks = mask_generator.generate(img)

        # Create a subdirectory for each image to store its masks
        image_mask_dir = output_dir / (image_name + "_masks")
        image_mask_dir.mkdir(parents=True, exist_ok=True)

        # Extract embeddings for each mask region
        mask_embeddings = []
        for idx, m in enumerate(masks):
            mask_array = (m["segmentation"].astype(np.uint8) * 255)
            mask_filename = f"{image_name}_mask_{idx}.png"
            mask_path = image_mask_dir / mask_filename

            # Save mask
            cv2.imwrite(str(mask_path), mask_array)

            # Record mask details
            mask_info_records.append({
                "Image_Name": image_name,
                "Species": species,
                "Mask_Index": idx,
                "Area": m["area"],
                "BBox": m["bbox"],
                "Predicted_IOU": m["predicted_iou"],
                "Stability_Score": m["stability_score"],
                "Mask_File": str(mask_path)
            })

            # Extract embedding from masked region:
            # Crop the image to the bounding box of the mask
            x, y, w, h = m["bbox"]
            crop = img[int(y):int(y + h), int(x):int(x + w), :]
            # Optionally apply the mask to focus on wing pixels only
            # We'll just use the bounding box crop for simplicity
            # If you want to mask out background, you can do:
            # crop[mask_array[int(y):int(y+h), int(x):int(x+w)) == 0] = (0,0,0)

            crop_tensor = tf.convert_to_tensor(crop, dtype=tf.uint8)
            emb = get_embedding_from_image(crop_tensor)
            mask_embeddings.append(emb)

        if len(mask_embeddings) > 0:
            # Average embeddings of all masks for this image
            image_emb = np.mean(mask_embeddings, axis=0)
        else:
            # If no masks, just a zero vector
            image_emb = np.zeros((1280,), dtype=np.float32)

        image_embeddings.append(image_emb)

    # Convert to numpy array
    image_embeddings = np.array(image_embeddings)
    print("Image embeddings shape:", image_embeddings.shape)

    # Save embeddings to a CSV
    emb_df = pd.DataFrame(image_embeddings)
    emb_df["Image_Name"] = image_names
    emb_df["Species"] = species_labels
    emb_df.to_csv(output_dir / "extracted_features.csv", index=False)

    # Save mask info
    mask_info_df = pd.DataFrame(mask_info_records)
    mask_info_df.to_csv(output_dir / "mask_details.csv", index=False)

    # PCA
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    features_pca = pca.fit_transform(image_embeddings)
    print("PCA-reduced features shape:", features_pca.shape)
    plot_pca_variance(pca, output_dir / "pca_variance_explained.png")

    # t-SNE
    tsne = TSNE(n_components=3, random_state=42, perplexity=10, n_iter=1000)
    features_tsne_3d = tsne.fit_transform(image_embeddings)
    features_tsne_2d = features_tsne_3d[:, :2]

    # Plots
    plot_embeddings_2d(features_pca[:, :2], image_names, "PCA Visualization (2D)",
                       output_path=output_dir / "pca_scatter_plot_2d.png",
                       species_labels=species_labels)

    plot_embeddings_3d(features_pca[:, :3], image_names, "PCA Visualization (3D)",
                       output_path=output_dir / "pca_scatter_plot_3d.png",
                       species_labels=species_labels)

    plot_embeddings_2d(features_tsne_2d, image_names, "t-SNE Visualization (2D)",
                       output_path=output_dir / "tsne_scatter_plot_2d.png",
                       species_labels=species_labels)

    plot_embeddings_3d(features_tsne_3d, image_names, "t-SNE Visualization (3D)",
                       output_path=output_dir / "tsne_scatter_plot_3d.png",
                       species_labels=species_labels)

    # Clustering
    clustering_pca = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, linkage='ward')
    cluster_labels_pca = clustering_pca.fit_predict(features_pca)

    clustering_tsne = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, linkage='ward')
    cluster_labels_tsne = clustering_tsne.fit_predict(features_tsne_3d)

    # Dendrogram
    linked = linkage(features_pca, method='ward')
    plot_dendrogram(linked, output_dir / "dendrogram.png", labels=image_names)

    clusters_df = pd.DataFrame({
        'Image_Name': image_names,
        'Species': species_labels,
        'Cluster_Label_PCA': cluster_labels_pca,
        'Cluster_Label_TSNE': cluster_labels_tsne
    })

    clusters_df.to_csv(output_dir / "cluster_labels_with_pca_tsne.csv", index=False)
    np.save(output_dir / "linkage_matrix.npy", linked)

    plot_cluster_distribution(cluster_labels_pca, output_dir / "pca_cluster_distribution.png")
    plot_cluster_distribution(cluster_labels_tsne, output_dir / "tsne_cluster_distribution.png")

    print("Cluster composition by species (PCA):")
    print(clusters_df.groupby('Cluster_Label_PCA')['Species'].value_counts())

    print("Cluster composition by species (t-SNE):")
    print(clusters_df.groupby('Cluster_Label_TSNE')['Species'].value_counts())

    print("Hierarchical clustering completed successfully.")


if __name__ == "__main__":
    main()
