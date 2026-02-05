import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# إعدادات واجهة المستخدم
st.set_page_config(page_title="كاشف الطفيليات المجهري", layout="centered")
st.title("🔬 نظام التشخيص الذكي للطفيليات")
st.info("قم برفع صورة من المجهر أو استخدم الكاميرا مباشرة")

# دالة البحث عن الملفات
def get_files():
    m = next((f for f in os.listdir() if f.endswith(".h5")), None)
    l = next((f for f in os.listdir() if f.endswith(".txt") and "req" not in f), None)
    return m, l

model_file, label_file = get_files()

@st.cache_resource
def load_model_safely(m_path, l_path):
    # تحميل النموذج مع إيقاف الترجمة لتجنب مشاكل التوافق
    model = tf.keras.models.load_model(m_path, compile=False)
    
    # حل مشكلة "2 input tensors": إذا كان النموذج مغلفاً، نأخذ الطبقة الداخلية
    if hasattr(model, 'layers') and len(model.layers) > 0:
        if isinstance(model.layers[0], tf.keras.Model):
            model = model.layers[0]

    with open(l_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

if model_file and label_file:
    try:
        model, class_names = load_model_safely(model_file, label_file)
        
        img_file = st.camera_input("التقط صورة العينة")
        if not img_file:
            img_file = st.file_uploader("أو ارفع صورة من الجهاز", type=['jpg', 'png', 'jpeg'])

        if img_file:
            image = Image.open(img_file).convert("RGB")
            st.image(image, caption="المعينة المختارة", use_container_width=True)
            
            # معالجة الصورة
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image).astype(np.float32) / 127.5 - 1
            data = np.expand_dims(img_array, axis=0)
            
            # التوقع باستخدام الطريقة المباشرة لتجنب أخطاء Keras 3
            prediction = model(data, training=False).numpy()
            index = np.argmax(prediction)
            percent = prediction[0][index] * 100
            
            # عرض النتيجة بوضوح
            st.success(f"النتيجة المتوقعة: {class_names[index]}")
            st.progress(int(percent))
            st.write(f"نسبة الثقة: {percent:.2f}%")

    except Exception as e:
        st.error(f"حدث خطأ فني: {e}")
        st.warning("تأكد من أنك رفعت ملف keras_model.h5 الأصلي وليس ملفاً مضغوطاً.")
else:
    st.error("لم يتم العثور على ملفات النموذج (.h5) أو الأسماء (.txt) في GitHub.")
