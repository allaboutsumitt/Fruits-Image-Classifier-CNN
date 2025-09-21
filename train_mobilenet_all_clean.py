# train_mobilenet_all_clean.py
import os, json, numpy as np, tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt

# ---------- Helper: find dataset root that contains 'Training' and 'Test' ----------
def find_data_root(start_dirs):
    for base in map(Path, start_dirs):
        if not Path(base).exists():
            continue
        # Quick checks
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
        # Deep search
        for p in Path(base).rglob("*"):
            if p.is_dir() and (p / "Training").is_dir() and (p / "Test").is_dir():
                return p
    return None

# ---------- Config ----------
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
K.set_image_data_format("channels_last")

IMG_SIZE = (128, 128)   # smaller than 224 → faster
BATCH = 16
EPOCHS_FE = 2           # feature extraction
EPOCHS_FT = 2           # fine-tuning

# Optional: quick dev mode (limits steps per epoch to iterate faster)
DEV_FAST = True
MAX_TRAIN_STEPS = 200
MAX_VAL_STEPS = 80

# Threading (optional; may help some CPUs)
try:
    import multiprocessing
    cores = multiprocessing.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(max(1, cores - 1))
    tf.config.threading.set_inter_op_parallelism_threads(max(1, cores // 2))
except Exception:
    pass

# ---------- Locate dataset ----------
DATA_ROOT = find_data_root([
    r"C:\Users\HP\Desktop\fruits-360",
    r"C:\Users\HP\Desktop\yes\fruits-360",
    r"C:\Users\HP\Desktop",
    Path.cwd(),
])
if DATA_ROOT is None:
    raise FileNotFoundError("Could not find 'Training' and 'Test'. Make sure the dataset is extracted.")

TRAIN_DIR = Path(DATA_ROOT) / "Training"
TEST_DIR  = Path(DATA_ROOT) / "Test"
print("Using DATA_ROOT:", DATA_ROOT)

# ---------- Build tf.data datasets ----------
def make_split_ds(root, subset, class_names=None, shuffle=True):
    return tf.keras.utils.image_dataset_from_directory(
        root,
        validation_split=0.2,
        subset=subset,      # "training" or "validation"
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="int",   # int labels → faster + less memory
        shuffle=shuffle,
        class_names=class_names,
    )

train_ds = make_split_ds(TRAIN_DIR, "training", shuffle=True)
class_names = train_ds.class_names
num_classes = len(class_names)
val_ds   = make_split_ds(TRAIN_DIR, "validation", class_names=class_names, shuffle=False)
test_ds  = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH, label_mode="int",
    shuffle=False, class_names=class_names
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

train_ds_fit, val_ds_fit = train_ds, val_ds
if DEV_FAST:
    train_ds_fit = train_ds.take(MAX_TRAIN_STEPS)
    val_ds_fit   = val_ds.take(MAX_VAL_STEPS)

# ---------- Model (clean; no Lambda) ----------
# Light augmentation
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
# Replace Lambda(preprocess_input) with native Rescaling to [-1, 1] (MobileNetV2 expects this)
x = layers.Rescaling(1.0/127.5, offset=-1.0, name="preproc")(x)

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
base.trainable = False  # Stage 1: feature extraction

x = base(x, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
model = models.Model(inputs, outputs, name="fruit_classifier_mobilenetv2_clean")

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1, min_lr=1e-6),
]

# Stage 1: feature extraction
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist_fe = model.fit(train_ds_fit, validation_data=val_ds_fit, epochs=EPOCHS_FE, callbacks=callbacks)

# Stage 2: fine-tune (unfreeze small tail; avoid BatchNorm)
unfreeze = 40
for l in base.layers[-unfreeze:]:
    if not isinstance(l, layers.BatchNormalization):
        l.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist_ft = model.fit(train_ds_fit, validation_data=val_ds_fit, epochs=EPOCHS_FT, callbacks=callbacks)

# ---------- Evaluate on Test ----------
y_true = np.concatenate([y.numpy() for _, y in test_ds])
probs = model.predict(test_ds, verbose=0)
y_pred = probs.argmax(axis=1)
acc = (y_true == y_pred).mean()
print(f"Test accuracy: {acc:.4f}")

rep = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(rep)
cm = confusion_matrix(y_true, y_pred)

# ---------- Save ----------
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

model_path = "models/fruit_mobilenetv2_all_clean.keras"
model.save(model_path)

# index -> class name
idx_to_class = {i: name for i, name in enumerate(class_names)}
with open("class_indices.json", "w", encoding="utf-8") as f:
    json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

with open("outputs/report_allclasses.txt", "w", encoding="utf-8") as f:
    f.write("Backbone: MobileNetV2 (clean, no Lambda)\n")
    f.write(f"IMG_SIZE: {IMG_SIZE}, BATCH: {BATCH}\n")
    f.write(f"Test accuracy: {acc:.4f}\n\n{rep}")

sns.set(font_scale=0.6 if num_classes > 50 else 1.0)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues", cbar=False)
plt.title("Confusion Matrix - MobileNetV2 (All Classes)")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.savefig("outputs/confusion_allclasses.png"); plt.close()

# Training curves
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

print("Saved:")
print("-", model_path)
print("- class_indices.json")
print("- outputs/report_allclasses.txt, confusion_allclasses.png, acc_*.png, loss_*.png")