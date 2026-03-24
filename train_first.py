import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


data_dir = r"C:\Users\Furkan\Desktop\Model_Deneme\tf_env\224" 
batch_size = 16 
img_height = 224
img_width = 224

# Verileri Yükleme ve %80 Eğitim, %20 Test Olarak Ayırma
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

class_names = train_ds.class_names # ['Normal', 'Scoliosis', 'Spondylolisthesis']

# Veri Artırma Katmanları
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Transfer Learning Modeli (EfficientNetB0)
base_model = EfficientNetB0(input_shape=(img_height, img_width, 3), 
                            include_top=False, 
                            weights='imagenet')
base_model.trainable = False # İlk deneme için önceden eğitilmiş ağırlıkları donduruyoruz

# Kendi Sınıflandırıcı Katmanlarımızı Ekleme
model = models.Sequential([
    tf.keras.Input(shape=(img_height, img_width, 3)),
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax', dtype='float32') # 3 sınıfımız var
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nEğitim başlıyor...")
epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

model.save("omurga_modeli.keras")
print("\n💾 Model başarıyla 'omurga_modeli.keras' olarak kaydedildi!\n")
print("📊 Modelin test verisi üzerindeki performansı hesaplanıyor, lütfen bekleyin...")

y_true = []
y_pred = []


for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))  
    y_true.extend(labels.numpy())           


precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
final_val_loss = history.history['val_loss'][-1]
final_val_acc = accuracy_score(y_true, y_pred)


print("\n" + "="*50)
print(f"{'🩺 YAPAY ZEKA MODEL KARNESİ':^50}")
print("="*50)
print(f" Toplam Epoch Sayısı : {epochs}")
print(f" Test Hata (Loss)    : {final_val_loss:.4f}")
print("-" * 50)
print(f" Accuracy (Doğruluk) : % {final_val_acc * 100:.2f}")
print(f" Precision (Kesinlik): % {precision * 100:.2f}")
print(f" Recall (Duyarlılık) : % {recall * 100:.2f}")
print(f" F1 Score (F1 Skoru) : % {f1 * 100:.2f}")
print("="*50 + "\n")