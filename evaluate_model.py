
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "PlantVillage"
MODEL_PATH = BASE_DIR / "models" / "plant_disease_multiplant_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

OUTPUT_DIR = BASE_DIR / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.20
RANDOM_SEED = 42


# --------------------------------------------------
# 2. CHECK REQUIRED FILES
# --------------------------------------------------

print("\n" + "=" * 55)
print("       PLANT DISEASE MODEL EVALUATION")
print("=" * 55)

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )


# --------------------------------------------------
# 3. LOAD CLASS NAMES
# --------------------------------------------------

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

if not isinstance(class_names, list) or not class_names:
    raise ValueError(
        "class_names.json must contain a non-empty list of class names."
    )

print(f"\nNumber of classes: {len(class_names)}")


# --------------------------------------------------
# 4. LOAD EXISTING TRAINED MODEL
# --------------------------------------------------

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# 5. LOAD VALIDATION DATASET
# --------------------------------------------------

print("\nLoading validation images...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    str(DATASET_DIR),
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=RANDOM_SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

print("Validation dataset loaded successfully.")


# --------------------------------------------------
# 6. GENERATE PREDICTIONS
# --------------------------------------------------

print("\nEvaluating model. Please wait...")

true_labels = []
predicted_labels = []

for images, labels in validation_dataset:
    predictions = model.predict(images, verbose=0)

    batch_predictions = np.argmax(predictions, axis=1)

    true_labels.extend(labels.numpy().astype(int))
    predicted_labels.extend(batch_predictions.astype(int))

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

if len(true_labels) == 0:
    raise ValueError("No validation images were found.")

if len(true_labels) != len(predicted_labels):
    raise ValueError("The number of predictions does not match the labels.")

if np.max(true_labels) >= len(class_names):
    raise ValueError("Dataset labels do not match class_names.json.")

print(f"Images evaluated: {len(true_labels)}")


# --------------------------------------------------
# 7. CALCULATE EVALUATION METRICS
# --------------------------------------------------

accuracy = accuracy_score(true_labels, predicted_labels)

precision = precision_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0,
)

recall = recall_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0,
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0,
)


# --------------------------------------------------
# 8. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 55)
print("             EVALUATION RESULTS")
print("=" * 55)

print(f"Total validation images : {len(true_labels)}")
print(f"Accuracy                : {accuracy * 100:.2f}%")
print(f"Weighted Precision      : {precision * 100:.2f}%")
print(f"Weighted Recall         : {recall * 100:.2f}%")
print(f"Weighted F1-Score       : {f1 * 100:.2f}%")

print("=" * 55)


# --------------------------------------------------
# 9. CLASSIFICATION REPORT
# --------------------------------------------------

report = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
)

print("\nCLASSIFICATION REPORT")
print("-" * 55)
print(report)

report_path = OUTPUT_DIR / "classification_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("PLANT DISEASE MODEL EVALUATION\n\n")
    file.write(f"Validation images: {len(true_labels)}\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    file.write(f"Weighted Precision: {precision * 100:.2f}%\n")
    file.write(f"Weighted Recall: {recall * 100:.2f}%\n")
    file.write(f"Weighted F1-Score: {f1 * 100:.2f}%\n\n")
    file.write("CLASSIFICATION REPORT\n\n")
    file.write(report)

print(f"\nClassification report saved to:\n{report_path}")


# --------------------------------------------------
# 10. CREATE CONFUSION MATRIX
# --------------------------------------------------

matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names))),
)

figure_size = max(12, len(class_names) * 0.9)

fig, ax = plt.subplots(figsize=(figure_size, figure_size))

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names,
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues",
    colorbar=False,
    values_format="d",
)

ax.set_title("Plant Disease Detection - Confusion Matrix")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")

plt.tight_layout()

matrix_path = OUTPUT_DIR / "confusion_matrix.png"

plt.savefig(
    matrix_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Confusion matrix saved to:\n{matrix_path}")


# --------------------------------------------------
# 11. SAVE METRICS
# --------------------------------------------------

metrics = {
    "model": MODEL_PATH.name,
    "validation_images": int(len(true_labels)),
    "number_of_classes": len(class_names),
    "accuracy_percent": round(accuracy * 100, 2),
    "weighted_precision_percent": round(precision * 100, 2),
    "weighted_recall_percent": round(recall * 100, 2),
    "weighted_f1_score_percent": round(f1 * 100, 2),
}

metrics_path = OUTPUT_DIR / "metrics.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print(f"Metrics saved to:\n{metrics_path}")


# --------------------------------------------------
# 12. COMPLETION MESSAGE
# --------------------------------------------------

print("\n" + "=" * 55)
print("          EVALUATION COMPLETED!")
print("=" * 55)

print("\nGenerated files:")
print("1. evaluation_results/classification_report.txt")
print("2. evaluation_results/confusion_matrix.png")
print("3. evaluation_results/metrics.json")

print("\nYour trained model has not been retrained or modified.")
