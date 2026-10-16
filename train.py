"""Train a 3-class road classifier (crack / normal / pothole) with MobileNetV2 transfer learning.

Expects data/processed/{train,val,test}/{crack,normal,pothole}/*.jpg  (see prepare_data.py).
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).parent
DATA = ROOT / "data" / os.environ.get("DATA_DIR", "processed")
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
IMG = (224, 224)
BATCH = int(os.environ.get("BATCH", 32))
SEED = 42
BACKBONE = os.environ.get("BACKBONE", "mobilenetv2")
TAG = os.environ.get("TAG", BACKBONE)
HEAD_EP, FT_EP = int(os.environ.get("HEAD_EP", 10)), int(os.environ.get("FT_EP", 15))
FT_LAYERS = int(os.environ.get("FT_LAYERS", 40))
if os.environ.get("MIXED", "1") == "1":
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
tf.keras.utils.set_random_seed(SEED)


def load(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        DATA / split, image_size=IMG, batch_size=BATCH, shuffle=shuffle, seed=SEED,
        label_mode="categorical")


train_ds, val_ds = load("train", True), load("val", False)
test_sets = {d.name: load(d.name, False) for d in sorted(DATA.glob("test*"))}
classes = train_ds.class_names
print("Classes:", classes)

# class weights to handle imbalance
labels = np.concatenate([np.argmax(y, 1) for _, y in train_ds])
cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(len(classes)), y=labels)))

AUTOTUNE = tf.data.AUTOTUNE
train_ds, val_ds = (d.prefetch(AUTOTUNE) for d in (train_ds, val_ds))

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.25),
    tf.keras.layers.RandomBrightness(0.2),
])

BACKBONES = {  # name -> (constructor, preprocess)  (EfficientNet models rescale internally: raw 0-255 input)
    "mobilenetv2": (tf.keras.applications.MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input),
    "efficientnetv2s": (tf.keras.applications.EfficientNetV2S, lambda x: x),
    "convnext_tiny": (tf.keras.applications.ConvNeXtTiny, lambda x: x),
}
ctor, preprocess = BACKBONES[BACKBONE]
base = ctor(input_shape=IMG + (3,), include_top=False, weights="imagenet")
base.trainable = False
inp = tf.keras.Input(IMG + (3,))
x = augment(inp)
x = preprocess(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(len(classes), activation="softmax", dtype="float32")(x)
model = tf.keras.Model(inp, out)
RESUME = os.environ.get("RESUME")
if RESUME:  # continue from a saved model: skip head training, fine-tune nested backbone
    model = tf.keras.models.load_model(MODELS / RESUME)
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))


def fit(epochs, lr, tag):
    model.compile(tf.keras.optimizers.Adam(lr), "categorical_crossentropy", metrics=["accuracy"])
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(MODELS / f"best_{TAG}.keras", monitor="val_accuracy",
                                           save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3),
    ]
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=cw, callbacks=cbs)


h1 = fit(HEAD_EP, 1e-3, "head") if not RESUME else None
base.trainable = True
for layer in base.layers[:-FT_LAYERS]:  # fine-tune the top blocks only
    layer.trainable = False
h2 = fit(FT_EP, float(os.environ.get("FT_LR", 1e-5)), "finetune")

model = tf.keras.models.load_model(MODELS / f"best_{TAG}.keras")
(MODELS / "classes.json").write_text(json.dumps(classes))

# evaluation on every test set
for tname, tds in test_sets.items():
    y_true = np.concatenate([np.argmax(y, 1) for _, y in tds])
    y_pred = np.argmax(model.predict(tds.prefetch(AUTOTUNE)), 1)
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print(tname); print(report)
    (OUT / f"report_{TAG}_{tname}.txt").write_text(report)
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.savefig(OUT / f"confusion_{TAG}_{tname}.png", dpi=150); plt.close()

hs = [h for h in (h1, h2) if h]
acc = sum((h.history["accuracy"] for h in hs), [])
vacc = sum((h.history["val_accuracy"] for h in hs), [])
plt.plot(acc, label="train"); plt.plot(vacc, label="val"); plt.legend(); plt.xlabel("epoch")
plt.ylabel("accuracy"); plt.savefig(OUT / f"accuracy_{TAG}.png", dpi=150)
