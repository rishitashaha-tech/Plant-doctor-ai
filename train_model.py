

from pathlib import Path
import tensorflow as tf
import json

DATASET_DIR = Path("PlantVillage")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = (160, 160)
BATCH_SIZE = 16
SEED = 42

# All 15 classes in a fixed order
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
]

# Check that every class folder exists
for class_name in CLASS_NAMES:
    folder = DATASET_DIR / class_name
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

print("Loading PlantVillage dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    class_names=CLASS_NAMES,
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    class_names=CLASS_NAMES,
    label_mode="int",
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

print("Loading MobileNetV2 transfer-learning model...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(160, 160, 3))

x = tf.keras.layers.Rescaling(
    1.0 / 127.5,
    offset=-1
)(inputs)

x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)

# 15 output classes instead of 3
outputs = tf.keras.layers.Dense(
    len(CLASS_NAMES),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=[early_stopping],
)

# Save as a new file so the old 3-class model stays safe
model.save(MODEL_DIR / "plant_disease_multiplant_model.keras")

# Save the class order used during training
with open(MODEL_DIR / "class_names.json", "w") as file:
    json.dump(CLASS_NAMES, file, indent=4)

print("\nTraining complete!")
print("New model: models/plant_disease_multiplant_model.keras")
print("Class names saved: models/class_names.json")
