"""
=============================================================
 FASHIONTINDER - STAGE 2: Colour-Aware Clothing Detector
=============================================================

GOAL:
  Detect clothing TYPE and dominant COLOUR from a real colour photo.

DATASET:
  clothing-dataset-small — already in data/
  ~5,000 colour JPEGs, 10 categories: dress, hat, longsleeve,
  outwear, pants, shirt, shoes, shorts, skirt, t-shirt.

HOW TO RUN:
  python stage2_detect.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.cluster import KMeans

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
DATA_DIR   = "data"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 10
MODEL_PATH = "stage2_model.keras"

# ---------------------------------------------------------------
# 1) LOAD DATA
# ---------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "validation"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (train_ds
            .map(lambda x, y: (preprocess(x), y))
            .cache().shuffle(1000).prefetch(AUTOTUNE))
val_ds   = (val_ds
            .map(lambda x, y: (preprocess(x), y))
            .cache().prefetch(AUTOTUNE))
test_ds  = (test_ds
            .map(lambda x, y: (preprocess(x), y))
            .cache().prefetch(AUTOTUNE))

# ---------------------------------------------------------------
# 2) BUILD MODEL  (MobileNetV2 transfer learning)
# ---------------------------------------------------------------
if os.path.exists(MODEL_PATH):
    print(f"\n>>> Loading saved model from {MODEL_PATH}\n")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # freeze ImageNet weights; only train the head

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes),
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # ---------------------------------------------------------------
    # 3) TRAIN
    # ---------------------------------------------------------------
    print("\n>>> Training...\n")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    model.save(MODEL_PATH)
    print(f">>> Model saved to {MODEL_PATH}")

_, val_acc  = model.evaluate(val_ds,  verbose=2)
_, test_acc = model.evaluate(test_ds, verbose=2)
print(f"\n>>> Validation accuracy: {val_acc:.2%}")
print(f">>> Test accuracy      : {test_acc:.2%}\n")

# ---------------------------------------------------------------
# 4) COLOUR EXTRACTION  (K-Means dominant colour)
# ---------------------------------------------------------------
COLOUR_PALETTE = {
    "black":  (0,   0,   0),
    "white":  (255, 255, 255),
    "gray":   (128, 128, 128),
    "red":    (220, 20,  20),
    "orange": (255, 140, 0),
    "yellow": (255, 215, 0),
    "green":  (34,  139, 34),
    "blue":   (30,  100, 220),
    "navy":   (10,  20,  80),
    "purple": (128, 0,   128),
    "pink":   (255, 105, 180),
    "brown":  (101, 67,  33),
    "beige":  (220, 200, 160),
}

def _nearest_colour(rgb):
    r, g, b = rgb
    best, best_d = "unknown", float("inf")
    for name, (pr, pg, pb) in COLOUR_PALETTE.items():
        d = (r - pr)**2 + (g - pg)**2 + (b - pb)**2
        if d < best_d:
            best, best_d = name, d
    return best

def dominant_colour(img_array, k=5):
    """
    img_array : H x W x 3, float32 in [0, 1]
    Drops near-white pixels (background), clusters the rest,
    returns the name of the most common cluster's colour.
    """
    pixels = (img_array.reshape(-1, 3) * 255).astype(np.float32)
    # Exclude likely background (near-white)
    mask = np.any(pixels < 230, axis=1)
    foreground = pixels[mask]
    if len(foreground) < k:
        foreground = pixels   # fallback: use everything

    km = KMeans(n_clusters=k, n_init=3, random_state=0)
    km.fit(foreground)
    counts = np.bincount(km.labels_)
    dominant = km.cluster_centers_[np.argmax(counts)]
    return _nearest_colour(dominant)

# ---------------------------------------------------------------
# 5) PREDICT ONE IMAGE  (change index to try others)
# ---------------------------------------------------------------
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# Load raw (un-preprocessed) val images just for display + colour extraction
raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "validation"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

for raw_images, labels in raw_val_ds.take(1):
    raw_img    = raw_images[0].numpy()          # [0, 255] for display + colour
    true_label = labels[0].numpy()

preprocessed = preprocess(raw_images[0:1])     # [-1, 1] for the model
predictions     = probability_model.predict(preprocessed, verbose=0)
predicted_class = np.argmax(predictions[0])
confidence      = np.max(predictions[0])
colour          = dominant_colour(raw_img / 255.0)

print(f">>> Clothing type  : {class_names[predicted_class]} ({confidence:.0%})")
print(f">>> Dominant colour: {colour}")
print(f">>> True label     : {class_names[true_label]}")

plt.figure(figsize=(4, 4))
plt.imshow(raw_img.astype(np.uint8))
plt.title(f"{class_names[predicted_class]} · {colour}\n({confidence:.0%} confidence)")
plt.axis("off")
plt.show()