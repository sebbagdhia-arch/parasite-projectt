import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# --- 1. إصلاح شامل لعيوب التوافق (DepthwiseConv2D) ---
class FixedDepthwise(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None) # التخلص من المعامل المسبب للخطأ
        super().__init__(*args, **kwargs)

# --- 2. إعداد الواجهة ---
st.set_page_config(page_title="كاشف الطفيليات 2026", layout="centered")
st.title("🔬 نظام التشخيص المجهري الآلي")
st.markdown("---")

@st.cache_resource
def load_model_miracle():
    # البحث عن الملفات
    m_path = next((f for f in os.listdir() if f.endswith(".h5")), None)
    l_path = next((f for f in os.listdir() if f.endswith(".txt") and "req" not in f.lower()), None)
    
    if not m_path or not l_path:
        return None, None
    
    # تحميل النموذج مع إجبار النظام على استخدام الطبقة المصلحة
    custom_obj = {'DepthwiseConv2D': FixedDepthwise}
    model = tf.keras.models.load_model(m_path, custom_objects=custom_obj, compile=False)
    
    with open(l_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
        
    return model, labels

# محاولة التحميل
try:
    model, class_names = load_model_miracle()
except Exception as e:
    st.error("⚠️ فشل النظام في التعرف على بنية النموذج القديم.")
    st.info("سنجرب الآن طريقة التحميل 'الخام'...")
    # محاولة تحميل احتياطية إذا فشلت الأولى
    model = tf.keras.models.load_model(next(f for f in os.listdir() if f.endswith(".h5")), compile=False)
    class_names = ["Parasite", "Normal"] # أسماء احتياطية

if model:
    source = st.camera_input("التقط صورة العينة")
    
    if source:
        # معالجة الصورة
        image = Image.open(source).convert("RGB")
        st.image(image, caption="تم التقاط العينة", use_container_width=True)
        
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image).astype(np.float32) / 127.5 - 1
        data = np.expand_dims(img_array, axis=0)
        
        # --- 3. حل مشكلة "2 Tensors" (الدوامة) ---
        # بدلاً من model.predict، نستخدم الاستدعاء المباشر كـ Tensor
        try:
            # تحويل البيانات إلى تنسور صريح لمنع Keras من إضافة "قناع" (Mask)
            input_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
            prediction = model(input_tensor, training=False).numpy()
            
            index = np.argmax(prediction)
            label = class_names[index]
            confidence = prediction[0][index]
            
            # عرض النتيجة
            st.success(f"النتيجة المتوقعة: **{label}**")
            st.metric("نسبة الدقة", f"{confidence*100:.2f}%")
            if confidence > 0.8: st.balloons()
            
        except Exception as e:
            st.error(f"خطأ في معالجة الصورة: {e}")
else:
    st.warning("يرجى التأكد من رفع ملفات .h5 و .txt بشكل سليم.")
