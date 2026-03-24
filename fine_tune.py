import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

data_dir = r"C:\Users\Furkan\Desktop\Model_Deneme\tf_env\224" 
batch_size = 16
img_height = 224
img_width = 224

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

# Eğittiğimiz modeli yüklüyoruz
print("Önceki model yükleniyor...")
model = tf.keras.models.load_model("omurga_modeli.keras")


# Modelin içindeki katmanları bulup EfficientNet'i eğitilebilir hale getiriyoruz
for layer in model.layers:
    if layer.name.startswith("efficientnet"):
        layer.trainable = True
        
print("EfficientNet kilitleri açıldı! Tüm ağırlıklar güncellenecek.")


model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

print("\nİnce ayar (Fine-Tuning) başlıyor...")
fine_tune_epochs = 10

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=fine_tune_epochs,
    callbacks=[early_stopping]
)

model.save("omurga_modeli_finetuned.keras")
print("\nİnce ayar tamamlandı ve yeni model başarıyla kaydedildi!\n")
print("İnce ayarlı modelin performansı hesaplanıyor, lütfen bekleyin...")

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

gercek_epoch_sayisi = len(history_fine.epoch)
final_val_loss = history_fine.history['val_loss'][-1]
final_val_acc = accuracy_score(y_true, y_pred)

print("\n" + "="*50)
print(f"{'🩺 İNCE AYAR (FINE-TUNING) KARNESİ':^50}")
print("="*50)
print(f" Gerçekleşen Epoch   : {gercek_epoch_sayisi} (Hedef: {fine_tune_epochs})")
print(f" Test Hata (Loss)    : {final_val_loss:.4f}")
print("-" * 50)
print(f" Accuracy (Doğruluk) : % {final_val_acc * 100:.2f}")
print(f" Precision (Kesinlik): % {precision * 100:.2f}")
print(f" Recall (Duyarlılık) : % {recall * 100:.2f}")
print(f" F1 Score (F1 Skoru) : % {f1 * 100:.2f}")
print("="*50 + "\n")
