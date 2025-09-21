import os, json, numpy as np, tensorflow as tf, random
from pathlib import Path
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Repro + data format
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
K.set_image_data_format("channels_last")

# Speed/settings
IMG_SIZE = (128, 128)
BATCH = 16
EPOCHS_FE = 2     # feature extraction epochs
EPOCHS_FT = 2     # fine-tuning epochs

# Train on half dataset? Two modes:
USE_HALF_DATASET = True        # turn on half-dataset training
HALF_BY = "images"             # "images" (recommended) or "classes"

# Fast iteration mode (also limits steps)
DEV_FAST = True
MAX_TRAIN_STEPS = 200
MAX_VAL_STEPS = 80

# Mixed precision (auto on GPU)
MIXED = False
if tf.config.list_physical_devices('GPU'):
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        MIXED = True
        print("Mixed precision: ON")
    except Exception:
        print("Mixed precision not available")
else:
    print("Running on CPU; enable GPU in Runtime > Change runtime type > GPU")

# Helper: find dataset root that contains 'Training' and 'Test'
def find_data_root(start_dirs):
    for base in map(Path, start_dirs):
        if not Path(base).exists():
            continue
        candidates = [
            base,
            base / "fruits-360",
            base / "Fruits-360",
            base / "fruits-360_dataset",
            base / "fruits-360_dataset" / "fruits-360",
            base / "archive",
            base / "archive" / "fruits-360",
        ]
        for cand in candidates:
            if (cand / "Training").is_dir() and (cand / "Test").is_dir():
                return cand
        for p in Path(base).rglob("*"):
            if p.is_dir() and (p / "Training").is_dir() and (p / "Test").is_dir():
                return p
    return None

# Look in /content and Drive
DATA_ROOT = find_data_root([
    r"C:\Users\HP\Desktop\fruits-360",
    r"C:\Users\HP\Desktop\yes\fruits-360",
    Path.cwd(),
])
if DATA_ROOT is None:
    raise FileNotFoundError("Could not find 'Training' and 'Test'. Link dataset to /content/fruits-360 or adjust paths.")
TRAIN_DIR = Path(DATA_ROOT) / "Training"
TEST_DIR  = Path(DATA_ROOT) / "Test"
print("Using DATA_ROOT:", DATA_ROOT)

# Build tf.data datasets
def make_split_ds(root, subset, class_names=None, shuffle=True):
    return tf.keras.utils.image_dataset_from_directory(
        root,
        validation_split=0.2,
        subset=subset,      # "training" or "validation"
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="int",
        shuffle=shuffle,
        class_names=class_names,
    )

# If HALF_BY == "classes", pick half the classes first
selected_class_names = None
if USE_HALF_DATASET and HALF_BY == "classes":
    # Build a temporary ds to read class names
    tmp_ds = make_split_ds(TRAIN_DIR, "training", shuffle=False)
    all_names = tmp_ds.class_names
    random.seed(SEED)
    random.shuffle(all_names)
    selected_class_names = sorted(all_names[: len(all_names)//2])  # 50% classes
    print(f"Using {len(selected_class_names)} classes out of {len(all_names)}.")

train_ds = make_split_ds(TRAIN_DIR, "training", shuffle=True,  class_names=selected_class_names)
class_names = train_ds.class_names
num_classes = len(class_names)
val_ds   = make_split_ds(TRAIN_DIR, "validation", shuffle=False, class_names=class_names)
test_ds  = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH, label_mode="int",
    shuffle=False, class_names=class_names
)

# tf.data performance
AUTOTUNE = tf.data.AUTOTUNE
opt = tf.data.Options()
opt.experimental_deterministic = False  # allow non-determinism for speed
train_ds = train_ds.with_options(opt).prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

# Limit steps per epoch (half-by-images and/or dev-fast)
def half_steps(ds):
    card = tf.data.experimental.cardinality(ds).numpy()
    # cardinality is in batches
    return max(1, card // 2)

train_steps_candidates = []
val_steps_candidates = []

if USE_HALF_DATASET and HALF_BY == "images":
    train_steps_candidates.append(half_steps(train_ds))
    val_steps_candidates.append(half_steps(val_ds))

if DEV_FAST:
    train_steps_candidates.append(MAX_TRAIN_STEPS)
    val_steps_candidates.append(MAX_VAL_STEPS)

steps_train = min(train_steps_candidates) if train_steps_candidates else None
steps_val   = min(val_steps_candidates) if val_steps_candidates else None

train_fit = train_ds.take(steps_train) if steps_train else train_ds
val_fit   = val_ds.take(steps_val) if steps_val else val_ds

print(f"Classes used: {num_classes}")
print(f"Steps/epoch (train): {steps_train if steps_train else 'full'} | (val): {steps_val if steps_val else 'full'}")

# Model (clean: no Lambda; uses Rescaling)
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ],
    name="augmentation",
)

inputs = layers.Input(shape=IMG_SIZE + (3,), name="image_rgb")
x = data_augmentation(inputs)
# MobileNetV2 expects [-1, 1]
x = layers.Rescaling(1.0/127.5, offset=-1.0, name="preproc")(x)

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
base.trainable = False  # Stage 1: freeze

x = base(x, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.4)(x)
dtype_last = "float32" if MIXED else None
outputs = layers.Dense(num_classes, activation="softmax", name="predictions", dtype=dtype_last)(x)
model = models.Model(inputs, outputs, name="fruit_classifier_mobilenetv2_clean")

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1, min_lr=1e-6),
]

# Stage 1: feature extraction
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
hist_fe = model.fit(train_fit, validation_data=val_fit, epochs=EPOCHS_FE, callbacks=callbacks)

# Stage 2: fine-tune small tail (avoid BatchNorm)
unfreeze = 40
for l in base.layers[-unfreeze:]:
    if not isinstance(l, layers.BatchNormalization):
        l.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
hist_ft = model.fit(train_fit, validation_data=val_fit, epochs=EPOCHS_FT, callbacks=callbacks)

# Evaluate on FULL test set (recommended)
y_true = np.concatenate([y.numpy() for _, y in test_ds])
probs = model.predict(test_ds, verbose=0)
y_pred = probs.argmax(axis=1)
acc = (y_true == y_pred).mean()
print(f"Test accuracy: {acc:.4f}")

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)
cm = confusion_matrix(y_true, y_pred)

# Save model + artifacts
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

MODEL_PATH = "models/fruit_mobilenetv2_half_clean.keras" if USE_HALF_DATASET else "models/fruit_mobilenetv2_all_clean.keras"
model.save(MODEL_PATH)
print("Saved model to:", MODEL_PATH)

idx_to_class = {i: name for i, name in enumerate(class_names)}
with open("class_indices.json", "w", encoding="utf-8") as f:
    json.dump(idx_to_class, f, ensure_ascii=False, indent=2)
print("Saved class_indices.json")

with open("outputs/report.txt", "w", encoding="utf-8") as f:
    f.write(f"Half mode: {USE_HALF_DATASET} by {HALF_BY}\n")
    f.write(f"IMG_SIZE: {IMG_SIZE}, BATCH: {BATCH}\n")
    f.write(f"Epochs: FE={EPOCHS_FE}, FT={EPOCHS_FT}\n")
    f.write(f"Steps/epoch: train={steps_train}, val={steps_val}\n")
    f.write(f"Test accuracy: {acc:.4f}\n\n")
    f.write(report)
print("Saved outputs/report.txt")

# Curves
def plot_and_save(hist, key, fname, title):
    plt.figure()
    plt.plot(hist.history.get(key, []), label=f"train {key}")
    plt.plot(hist.history.get(f"val_{key}", []), label=f"val {key}")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(fname); plt.close()

plot_and_save(hist_fe, "accuracy", "outputs/acc_fe.png", "Accuracy (Feature Extraction)")
plot_and_save(hist_fe, "loss",     "outputs/loss_fe.png","Loss (Feature Extraction)")
plot_and_save(hist_ft, "accuracy", "outputs/acc_ft.png", "Accuracy (Fine-tuning)")
plot_and_save(hist_ft, "loss",     "outputs/loss_ft.png","Loss (Fine-tuning)")
print("Saved training curves to outputs/acc_*.png and outputs/loss_*.png")