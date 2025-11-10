# Autoencoder + Clustering Pipeline for Insect Morphology Analysis

This repository provides a complete workflow for **image preprocessing**, **deep feature extraction**, and **unsupervised clustering** for large-scale insect morphology studies.  
The pipeline standardizes raw images, learns compact latent “fingerprints” using a convolutional autoencoder, and then groups similar specimens through clustering methods such as OPTICS, KMeans, and Agglomerative clustering.

This framework was developed in collaboration with biodiversity research teams and is designed for **reproducibility**, **scalability**, and **biological interpretability**.


<p align="center">
  <img src="img/model.png" width="90%">
</p>

---

## ✨ Features

- **Unified two-step pipeline**
  - **Preprocessing**: format unification, resizing, renaming.
  - **Autoencoder representation learning**: compact latent embeddings.
  - **Clustering**: OPTICS, KMeans, Agglomerative, with metrics and visualizations.

- **Rich visual outputs**
  - Activation maps  
  - Attention / importance heatmaps  
  - t-SNE / PCA projections  
  - Clustering diagnostics

- **Reproducible model + feature outputs**
  - Embedding files  
  - Model checkpoints  
  - Excel summary reports  

---

## 📦 Requirements

Python **3.9+**

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📁 Repository Structure

```
├── preprocess.py           # Preprocessing raw images
├── main.py                 # Autoencoder + clustering pipeline
├── user_guide.docx         # Full detailed documentation
├── README.md               # Project introduction
└── (Output folders created after running the scripts)
```

---

# Part 1 — Preprocessing (`preprocessing.py`)

<p align="center">
  <img src="img/preprocessing.png" width="90%">
</p>


### 🔧 What it does
- Scans all subfolders for images  
- Resizes & converts images to PNG  
- Standardizes naming scheme  
- Produces dataset statistics & label summaries  

### 🔑 Key Settings

| Setting | Meaning | Suggested |
|--------|---------|-----------|
| `input_dir` | Raw image folder | Your dataset path |
| `output_dir` | Output folder | e.g., `D:\preprocessing` |
| `IMAGE_SIZE` | Final image size | Match training resolution |
| `SUPPORTED_FORMATS` | Allowed types | png, jpg, jpeg, bmp, tif |

### ▶️ Run

```bash
python preprocess.py
```

### 📤 Outputs
- Resized PNGs  
- `statistics.csv`  
- `family_counts.csv`, `species_counts.csv`  
- `unique_families.txt`, `unique_species.txt`, `unique_names.txt`  
- `errors.log`  

---

# Part 2 — Autoencoder + Clustering (`main.py`)

### 🎯 Goal
Train an autoencoder to extract morphological features, then cluster them to reveal structure in the dataset.


### 🧠 Model Structure
A convolutional autoencoder compresses each image into a low-dimensional latent vector.  
These embeddings are clustered and visualized using multiple methods.

<p align="center">
  <img src="img/model_01.png" width="90%">
</p>

### 🔑 Hyperparameters

| Param | Location | Description | Recommendation |
|-------|----------|-------------|----------------|
| `data_folder` | in `main()` | Input processed images | Use preprocess output |
| `OUTPUT_DIR` | top of file | Save path | New folder per run |
| `IMG_SIZE` | top of file | Training size | 128–224 |
| `LATENT_DIM` | top of file | Embedding length | 32–128 |
| `BATCH_SIZE` | top of file | Step size | 8–32 |
| `EPOCHS` | top of file | Training epochs | 40–100 |
| `REG_FACTOR` | top of file | Weight decay | 1e-5 → 5e-5 |

### ▶️ Run

```bash
python main.py
```

---

# 📂 Output Directory Overview

All results are saved under your chosen `OUTPUT_DIR`.

### 🔍 Main Files & Their Use

| File/Folder | Description | Use |
|-------------|-------------|-----|
| `autoencoder.h5` | Full model | Reconstruction / further training |
| `encoder.h5` | Encoder-only | Extract embeddings for new images |
| `image_embeddings.txt` | Latent vectors | Clustering & ML |
| `activations/` | Filter activations | Interpret model focus |
| `heatmaps/` | Attention overlays | Morphological relevance |
| `tsne_plots/` | 2D visualizations | Inspect global structure |
| `clustering_results.xlsx` | Cluster labels & coordinates | Main summary |
| `clustering_metrics.xlsx` | Silhouette, CH, DBI | Compare cluster quality |
| `clustering_metrics.png` | Bar chart | Quick inspection |

---

# 📘 How to Interpret Results

### `clustering_results.xlsx`
Contains:
- filename  
- family / species / name  
- cluster labels (OPTICS, KMeans, Agglomerative)  
- t-SNE / PCA 2D coordinates  

### `clustering_metrics.xlsx`
- `silhouette` → higher = better  
- `calinski_harabasz` → higher = better  
- `davies_bouldin` → lower = better  

---

# 🔧 Hyperparameter Tips

1. Start small: `128×128`, `LATENT_DIM=64`, `BATCH_SIZE=16`, `EPOCHS=40`.  
2. Blurry reconstructions → increase `LATENT_DIM` or `EPOCHS`.  
3. Noisy heatmaps → increase `REG_FACTOR`.  
4. Compare clustering metrics to choose the best run.  
5. Use a fresh `OUTPUT_DIR` per experiment.

---

# 👀 Visual Outputs

- **Activation maps** → internal filter responses
<p align="center">
  <img src="img/Condylognatha-Achilidae_01_activations.png" width="80%">
</p>
  
- **Heatmaps** → important morphological regions

<p align="center">
  <img src="img/heatmap.png" width="80%">
</p>

- **t-SNE & PCA** → visual grouping of specimens  


---

# ❗ Troubleshooting

- No images found → check `data_folder`  
- Out of memory → reduce `IMG_SIZE` or `BATCH_SIZE`  
- OPTICS returns many `-1` → try KMeans/Agglomerative  
- Windows path issues → use double `\\` (e.g., `D:\\data\\images`)  

---

# 📄 Citation

If you use this pipeline in research, please cite:


---

# 📜 License

MIT License  

---

# 🙋 Contact

For questions, suggestions, or collaborations, please feel free to open an issue or contact the author shihan.guan@univ-rennes.fr.
