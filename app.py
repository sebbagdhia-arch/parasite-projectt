import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# --- 1. حل مشكلة التوافق مع تحديثات Keras الجديدة ---
import keras
# نقوم بتعديل وظيفة الطبقة المعقدة لتعمل مع النماذج القديمة
if hasattr(keras.layers, 'DepthwiseConv2D'):
    orig_init = keras.layers.DepthwiseConv2D.__init__
    def new_init(self, *args, **kwargs):
        kwargs.pop('groups', None) # حذف المعامل المسبب للمشاكل
        orig_init(self, *args, **kwargs)
    keras.layers.DepthwiseConv2D.__init__ = new_init

# --- 2. إعدادات الصفحة ---
st.set_page_config(page_title="كاشف الطفيليات", layout="centered")
st.title("🔬 نظام التمييز الآلي للطفيليات")
st.write("---")

# --- 3. البحث عن الملفات والتحقق منها ---
def find_files():
    m_file = next((f for f in os.listdir() if f.endswith(".h5")), None)
    l_file = next((f for f in os.listdir() if f.endswith(".txt") and f != "requirements.txt"), None)
    return m_file, l_file

model_path, label_path = find_files()

# فحص سلامة الملف قبل التحميل
if model_path:
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    if file_size_mb < 1:
        st.error(f"⚠️ ملف النموذج تالف أو غير مكتمل! الحجم الحالي: {file_size_mb:.2f} MB (يجب أن يكون أكبر من 2 MB).")
        st.stop()

@st.cache_resource
def load_my_model(m_path, l_path):
    # تحميل النموذج بدون ترجمة لتجنب الأخطاء
    model = tf.keras.models.load_model(m_path, compile=False)
    with open(l_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

if model_path and label_path:
    try:
        model, class_names = load_my_model(model_path, label_path)
        
        # واجهة الكاميرا
        source = st.camera_input("التقط صورة للعينة المجهرية")
        
        if source:
            # عرض الصورة
            image = Image.open(source).convert("RGB")
            st.image(image, caption="تم التقاط الصورة", use_container_width=True)
            
            # معالجة الصورة
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image).astype(np.float32) / 127.5 - 1
            data = np.expand_dims(img_array, axis=0)
            
            # --- 4. التوقع (الحل النهائي لمشكلة الإدخال) ---
            # نستخدم model(...) بدلاً من model.predict لتجنب تكرار الإدخال
            prediction_tensor = model(tf.constant(data), training=False)
            prediction = prediction_tensor.numpy()
            
            index = np.argmax(prediction)
            label_text = class_names[index]
            
            # تنظيف النص (إزالة الأرقام في البداية إن وجدت)
            if " " in label_text:
                label_text = label_text.split(" ", 1)[1]

            confidence = prediction[0][index]
            
            st.success(f"النتيجة: **{label_text}**")
            st.metric(label="درجة الثقة (الدقة)", value=f"{confidence*100:.2f}%")
            
    except Exception as e:
        st.error(f"حدث خطأ غير متوقع: {e}")
        st.info("جرب عمل Reboot للتطبيق من القائمة.")
else:
    st.warning("⚠️ يرجى التأكد من رفع ملفات .h5 و .txt بشكل صحيح.")
