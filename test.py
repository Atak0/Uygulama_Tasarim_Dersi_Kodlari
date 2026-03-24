import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Sınıf isimleri
class_names = ['Normal', 'scol', 'spond']

print("Model yükleniyor, lütfen bekleyin...")
model = tf.keras.models.load_model("omurga_modeli_dengeli.keras")

# Test etmek istediğin röntgen görüntüsünün yolunu buraya yaz
test_image_path = r"C:\Users\Furkan\Desktop\ornek_resim_224x224.jpg"

# Görüntüyü modele uygun hale getirme (Ön işleme)
img = image.load_img(test_image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) 

# 5. Tahmin yaptırma
predictions = model.predict(img_array)
score = predictions[0] # Modelimizin son katmanı zaten softmax olduğu için direkt olasılıkları verir

predicted_class = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

print("\n" + "="*30)
print("🩺 TAHMİN SONUCU")
print("="*30)
print(f"Modelin Teşhisi : {predicted_class}")
print(f"Eminlik Oranı   : %{confidence:.2f}")
print("-"*30)
print("Tüm Olasılıklar:")
for i, sinif in enumerate(class_names):
    print(f"- {sinif}: %{score[i]*100:.2f}")