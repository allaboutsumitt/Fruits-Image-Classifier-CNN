# train_all_fast.py
import os, json, numpy as np, tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt
from data_finder import find_data_root

# -------- Speed-oriented settings --------
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
K.set_image_data_format('channels_last')

IMG_SIZE = (160, 160)      # smaller = faster. You can try (128, 128) for even more speed
BATCH = 16                 # keep memory low
EPOCHS_FE = 3              # feature extraction epochs
EPOCHS_FT = 3              # fine-tuning epochs

DEV_FAST = True            # True = limit steps per epoch for quick iterations
MAX_TRAIN_STEPS = 200      # number of batches per epoch in dev mode
MAX_VAL_STEPS = 80

# Optional: tune CPU threading (can help on some CPUs)
try:
    import multiprocessing
    cores = multiprocessing.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(max(1, cores - 1))
    tf.config.threading.set_inter_op_parallelism_threads(max(1, cores // 2))
except Exception:
    pass

# Mixed precision (only if a GPU is present)
if tf.config.list_physical_devices('GPU'):
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        MIXED = True
    except Exception:
        MIXED = False
else:
    MIXED = False

# 1) Locate dataset (folder that contains 'Training' and 'Test')
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

# 2) Build tf.data datasets (fast input pipeline)
def make_ds(root, subset=None, shuffle=True, class_names=None):
    return tf.keras.utils.image_dataset_from_directory(
        root,
        validation_split=0.2 if subset in ("training","validation") else None,
        subset=subset,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode='int',
        shuffle=shuffle,
        class_names=class_names,
    )

train_ds = make_ds(TRAIN_DIR, subset="training", shuffle=True)
class_names = train_ds.class_names
num_classes = len(class_names)
val_ds   = make_ds(TRAIN_DIR, subset="validation", shuffle=False, class_names=class_names)
test_ds  = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH, label_mode='int',
    shuffle=False, class_names=class_names
)

AUTOTUNE = tf.data.AUTOTUNE

# Prefer prefetch for CPU; cache to disk is optional but large. Leave .cache() off unless you have fast SSD space.
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

# Limit steps per epoch for faster dev iterations
if DEV_FAST:
    train_ds_fit = train_ds.take(MAX_TRAIN_STEPS)
    val_ds_fit   = val_ds.take(MAX_VAL_STEPS)
else:
    train_ds_fit, val_ds_fit = train_ds, val_ds

# 3) Data augmentation layers (light to keep it fast)
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ],
    name="augmentation",
)

# 4) Build MobileNetV2 transfer model (fast and accurate)
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
base.trainable = False  # Stage 1: feature extraction

inputs = layers.Input(shape=IMG_SIZE + (3,), name="image_rgb")
x = data_augmentation(inputs)
# Use MobileNetV2 preprocess: scale to [-1, 1]
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mnet_preproc
x = layers.Lambda(mnet_preproc, name="preproc")(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.4)(x)
# Keep final layer in float32 if mixed precision is on
dtype_last = "float32" if MIXED else None
outputs = layers.Dense(num_classes, activation="softmax", name="predictions", dtype=dtype_last)(x)
model = models.Model(inputs, outputs, name="fruit_classifier_mobilenetv2")

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=1, min_lr=1e-6),
]

# 5) Train - Stage 1 (feature extraction)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',  # int labels = faster + less memory than one-hot
    metrics=['accuracy'],
)
hist_fe = model.fit(train_ds_fit, validation_data=val_ds_fit, epochs=EPOCHS_FE, callbacks=callbacks)

# 6) Train - Stage 2 (fine-tune top layers)
# Unfreeze a small tail of layers (avoid BatchNorm)
unfreeze = 40
for l in base.layers[-unfreeze:]:
    if not isinstance(l, layers.BatchNormalization):
        l.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
hist_ft = model.fit(train_ds_fit, validation_data=val_ds_fit, epochs=EPOCHS_FT, callbacks=callbacks)

# 7) Evaluate on Test
y_true = []
for _, y in test_ds:
    y_true.append(y.numpy())
y_true = np.concatenate(y_true)
probs = model.predict(test_ds)
y_pred = probs.argmax(axis=1)
acc = (y_true == y_pred).mean()
print(f"Test accuracy: {acc:.4f}")

from sklearn.metrics import classification_report, confusion_matrix
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)
cm = confusion_matrix(y_true, y_pred)

# 8) Save everything
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

model_path = "models/fruit_allclasses_tfdata_mnet.keras"
model.save(model_path)

idx_to_class = {i: name for i, name in enumerate(class_names)}
with open("class_indices.json", "w", encoding="utf-8") as f:
    json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

with open("outputs/report_allclasses.txt", "w", encoding="utf-8") as f:
    f.write(f"Backbone: MobileNetV2\n")
    f.write(f"IMG_SIZE: {IMG_SIZE}, BATCH: {BATCH}\n")
    f.write(f"Test accuracy: {acc:.4f}\n\n")
    f.write(report)

sns.set(font_scale=0.6 if len(class_names) > 50 else 1.0)
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