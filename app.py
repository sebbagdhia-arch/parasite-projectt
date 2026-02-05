import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# --- الخطوة الجراحية: إصلاح Keras 3 يدوياً قبل أي شيء ---
import keras
from keras.src.models.functional import Functional
from keras.src.models.sequential import Sequential

# دالة سحرية لإجبار النموذج على استقبال مدخل واحد فقط
def patched_call(self, inputs, *args, **kwargs):
    if isinstance(inputs, list) and len(inputs) > 1:
        inputs = inputs[0]  # خذ الصورة فقط وتجاهل القناع (Mask)
    return self._old_call(inputs, *args, **kwargs)

# تطبيق الترقيع على الطبقات الوظيفية
if not hasattr(Functional, '_old_call'):
    Functional._old_call = Functional.call
    Functional.call = patched_call

# --- إعداد الواجهة ---
st.set_page_config(page_title="كاشف الطفيليات المجهري", layout="centered")
st.title("🔬 مختبر التشخيص الذكي (إصدار 2026)")

@st.cache_resource
def load_everything():
    model_path = next((f for f in os.listdir() if f.endswith(".h5")), None)
    label_path = next((f for f in os.listdir() if f.endswith(".txt") and "req" not in f.lower()), None)
    
    if not model_path or not label_path:
        return None, None
    
    # تحميل النموذج (الترقيع أعلاه سيجعله يعمل الآن)
    model = tf.keras.models.load_model(model_path, compile=False)
    
    with open(label_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]
    
    return model, class_names

# تشغيل التحميل
try:
    model, class_names = load_everything()
except Exception as e:
    st.error(f"حدث خطأ في تحميل المحرك: {e}")
    st.info("جرب عمل Reboot من القائمة الجانبية.")
    st.stop()

if model and class_names:
    img_file = st.camera_input("التقط صورة العينة")
    
    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="الصورة تحت التحليل", use_container_width=True)
        
        # تجهيز الصورة
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image).astype(np.float32) / 127.5 - 1
        data = np.expand_dims(img_array, axis=0)
        
        # التوقع
        with st.spinner('جاري الفحص المجهري...'):
            prediction = model.predict(data, verbose=0)
            index = np.argmax(prediction)
            confidence = prediction[0][index]
            
            # تنظيف الاسم
            label = class_names[index]
            clean_label = label.split(" ", 1)[1] if " " in label else label
            
            st.success(f"النتيجة: **{clean_label}**")
            st.metric("نسبة التأكد", f"{confidence*100:.2f}%")
            if confidence > 0.7: st.balloons()
else:
    st.warning("⚠️ النظام بانتظار رفع ملفات النموذج (.h5) والأسماء (.txt)")
