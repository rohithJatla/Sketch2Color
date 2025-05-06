# Sketch2Color using GAN

**Authors**

- Rohith Jatla (`jatlar1`)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Approach](#approach)

   - [Data Preparation](#data-preparation)
   - [Model Design](#model-design)
   - [Loss Functions & Training](#loss-functions--training)

3. [Directory Layout](#directory-layout)
4. [Getting Started](#getting-started)
5. [Usage](#usage)
6. [Results](#results)
7. [Future Work](#future-work)
8. [License](#license)

---

## Project Overview

This project implements a **conditional Generative Adversarial Network (GAN)**—specifically the **Pix2Pix** framework—to learn a mapping from line‑drawing “outline” inputs to fully‑colored landscape or anime‑style images. By training on paired image halves (left = outline, right = color), our model learns both low‑level texture and high‑level semantic color distributions, enabling realistic colorization of new outlines.

---

## Approach

### Data Preparation

1. **Collect & Organize**

   - Place your full‑color JPEG images in `data/train/` (training set) and a few examples in `data/test/` (demo set).

2. **Automatic Splitting**

   - Each image is split vertically in half:

     - **Left half** → **input** (outline/grayscale)
     - **Right half** → **target** (ground‑truth color)

3. **Normalization & Batching**

   - Pixel values are scaled from `[0,255]` → `[-1,+1]`.
   - Data is batched and shuffled with TensorFlow’s `tf.data` API for efficient GPU training.

### Model Design

- **Generator (U‑Net)**

  - An **encoder–decoder** with 8 down‑sampling blocks (`Conv2D → BatchNorm → LeakyReLU`) and 8 up‑sampling blocks (`Conv2DTranspose → BatchNorm → ReLU`).
  - **Skip connections** between mirrored encoder/decoder layers preserve fine-grained outline details.

- **Discriminator (PatchGAN)**

  - Classifies **70×70** patches as real or fake, encouraging high‐frequency correctness.
  - Architecture: 4 convolutional layers with increasing filters, using `LeakyReLU` and **no** final activation (logits).

### Loss Functions & Training

- **Adversarial Loss**

  - Generator tries to **fool** the discriminator; discriminator learns to distinguish real vs. fake.

- **L1 Loss (Pixel)**

  - `L1 = mean(|target – generated|)` encourages color outputs close to the ground truth and reduces blurring.

- **Total Generator Loss**

  ```
  L_G = L_adv(generator) + λ * L1(generator, target)
  ```

  with λ = 100 for strong pixel‐level fidelity.

- **Optimization**

  - Both G and D use the **Adam** optimizer (lr=2e‑4, β1=0.5).
  - Trained for 150 epochs; checkpoints saved every 5 epochs.

---

## Directory Layout

```
Sketch2Color/
├── data/
│   ├── landscape/                ← full‑color input images
│   └── test_plus_landscapes/                 ← demo images
├── notebooks/
│   └── Sketch2Color.ipynb
├── models/
│   └── AnimeColorizationModelv1.h5
├── requirements.txt          ← pinned package versions
└── README.md
```

---

## Getting Started

1. **Clone the repo**

   ```bash
   git clone https://github.com/<your‑username>/Sketch2Color.git
   cd Sketch2Color
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data**

   - Drop your JPEGs into `data/train/` and `data/test/`.
   - Ensure each folder has at least 50 training images and 5 test images.

---

## Usage

1. **Launch Jupyter**

   ```bash
   jupyter notebook notebooks/Sketch2Color.ipynb
   ```

2. **Clear existing outputs** (`Kernel → Restart & Clear Output`), then **Run All**.
3. **Monitor training**

   - Loss curves and image grids appear inline.
   - Checkpoints in `models/`.

4. **Generate on new outlines**

   ```python
   from tensorflow import keras
   model = keras.models.load_model('models/AnimeColorizationModelv1.h5', compile=False)
   colorized = model.predict(your_outline_batch)
   ```

---

## Results

At epoch 150, the generator produces crisp, realistic colorizations that closely match ground truth:

![alt text](image.png)

---

## Future Work

- **Larger Dataset**: train on thousands of diverse scenes for better generalization.
- **Perceptual Loss**: integrate a VGG‑based feature loss for sharper textures.
- **Interactive Demo**: deploy via TensorFlow\.js or Flask for real‑time web colorization.
- **Alternate Architectures**: experiment with ResNet‑based or attention‑augmented generators.

---

## License

This project is licensed under the **MIT License**. See LICENSE for details.
