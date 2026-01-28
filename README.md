# =========================
# 1. Imports
# =========================
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# =========================
# 2. Paths & Config
# =========================
BASE_PATH = r"C:\Users\saptg\anaconda_projects\0059300d-d57a-4ccf-9060-af72d9236304\dataset\DATASET\DATASET"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 1   # baseline (1 epoch is enough)

# =========================
# 3. Data Generators
# =========================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    os.path.join(BASE_PATH, "TRAIN"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_test_gen.flow_from_directory(
    os.path.join(BASE_PATH, "VALIDATION"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = val_test_gen.flow_from_directory(
    os.path.join(BASE_PATH, "TEST"),
    target_size=IMG_SIZE,
    batch_size=1,
    shuffle=False,
    class_mode="categorical"
)

# =========================
# 4. Model (VGG16 + Head)
# =========================
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)
base_model.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(2, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# 5. Training
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# =========================
# 6. Save Training Curve (PNG)
# =========================
plt.figure(figsize=(6,4))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.title("Training Accuracy")
plt.legend()
plt.savefig("accuracy.png")
plt.close()

print("Saved accuracy.png")

# =========================
# 7. Evaluation (TEST SET)
# =========================
pred = model.predict(test_data)
y_pred = np.argmax(pred, axis=1)
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=test_data.class_indices.keys(),
    yticklabels=test_data.class_indices.keys()
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

print(classification_report(y_true, y_pred))

# =========================
# 8. Save Model
# =========================
model.save("waste_classifier_vgg16_final.keras")
model.save("waste_classifier_vgg16_final.h5")

print("Model saved (keras + h5)")
