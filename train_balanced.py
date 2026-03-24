import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

data_dir = r"C:\Users\Furkan\Desktop\Model_Deneme\tf_env\224_Dengeli"
batch_size = 16
img_height = 224
img_width = 224

# Verileri Yükleme (%80 Eğitim, %20 Test)
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

# Transfer Learning (EfficientNetB0)
base_model = EfficientNetB0(input_shape=(img_height, img_width, 3), 
                            include_top=False, 
                            weights='imagenet')
base_model.trainable = False 

model = models.Sequential([
    tf.keras.Input(shape=(img_height, img_width, 3)),
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3), # Verimiz arttığı için dropout'u %30'a çıkardık
    layers.Dense(3, activation='softmax', dtype='float32') 
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = callbacks.ModelCheckpoint(
    "omurga_modeli_dengeli.keras",
    save_best_only=True,      
    monitor='val_accuracy',   
    mode='max',              
    verbose=1                
)

print("\nGüçlendirilmiş veri seti ile eğitim başlıyor...")
epochs = 15

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint]
)

print("\n🎯 Eğitim bitti! En başarılı model 'omurga_modeli_dengeli.keras' olarak kaydedildi.")
print("\n📊 En başarılı model diskten yükleniyor ve karne hesaplanıyor...")

best_model = tf.keras.models.load_model("omurga_modeli_dengeli.keras")

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = best_model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
final_val_acc = accuracy_score(y_true, y_pred)

print("\n" + "="*50)
print(f"{'🩺 DENGELİ VERİ SETİ MODEL KARNESİ':^50}")
print("="*50)
print(f" Eğitimdeki Hedef Epoch : {epochs}")
print("-" * 50)
print(f" Accuracy (Doğruluk)    : % {final_val_acc * 100:.2f}")
print(f" Precision (Kesinlik)   : % {precision * 100:.2f}")
print(f" Recall (Duyarlılık)    : % {recall * 100:.2f}")
print(f" F1 Score (F1 Skoru)    : % {f1 * 100:.2f}")
print("="*50 + "\n")