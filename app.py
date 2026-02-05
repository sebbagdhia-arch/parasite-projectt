import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import keras

# --- 1. إصلاح مشكلة DepthwiseConv2D (للنماذج القديمة) ---
# هذا الجزء يمنع ظهور خطأ groups=1
if hasattr(keras.layers, 'DepthwiseConv2D'):
    orig_init = keras.layers.DepthwiseConv2D.__init__
    def new_init(self, *args, **kwargs):
        kwargs.pop('groups', None)
        orig_init(self, *args, **kwargs)
    keras.layers.DepthwiseConv2D.__init__ = new_init

# --- 2. إعداد الصفحة ---
st.set_page_config(page_title="كاشف الطفيليات", layout="centered")
st.title("🔬 نظام التمييز الآلي للطفيليات")

# --- 3. البحث عن الملفات ---
def find_files():
    m_file = next((f for f in os.listdir() if f.endswith(".h5")), None)
    l_file = next((f for f in os.listdir() if f.endswith(".txt") and f != "requirements.txt"), None)
    return m_file, l_file

model_path, label_path = find_files()

# --- 4. تحميل النموذج ---
@st.cache_resource
def load_my_model(m_path, l_path):
    # compile=False مهم جداً لتجنب أخطاء التدريب
    model = tf.keras.models.load_model(m_path, compile=False)
    with open(l_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

# --- 5. دالة التوقع الذكية (الحل لمشكلتك) ---
def smart_predict(model, data):
    try:
        # المحاولة 1: الطريقة القياسية
        return model.predict(data)
    except Exception:
        # المحاولة 2: إذا فشلت، نستخرج المحرك الداخلي (Functional)
        # هذا يتخطى خطأ "2 input tensors" الشهير
        # Teachable Machine يضع النموذج غالباً في الطبقة رقم 0
        try:
            return model.layers[0](data, training=False).numpy()
        except:
            # المحاولة 3: استدعاء النموذج مباشرة كـ دالة
            return model(data, training=False).numpy()

if model_path and label_path:
    # التحقق من حجم الملف لضمان عدم تلفه
    if os.path.getsize(model_path) / (1024 * 1024) < 1:
        st.error("⚠️ ملف النموذج تالف (صغير جداً). يرجى حذفه وإعادة رفعه بتمهل.")
        st.stop()

    try:
        model, class_names = load_my_model(model_path, label_path)
        
        source = st.camera_input("التقط صورة للعينة")
        
        if source:
            image = Image.open(source).convert("RGB")
            st.image(image, caption="تم التقاط الصورة", use_container_width=True)
            
            # تجهيز الصورة
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image).astype(np.float32) / 127.5 - 1
            data = np.expand_dims(img_array, axis=0)
            
            # تنفيذ التوقع باستخدام الدالة الذكية
            prediction = smart_predict(model, data)
            
            # عرض النتائج
            index = np.argmax(prediction)
            label_text = class_names[index]
            confidence = prediction[0][index]
            
            # إزالة الأرقام من الاسم (مثل "0 Parasite")
            if " " in label_text:
                label_text = label_text.split(" ", 1)[1]
            
            st.success(f"النتيجة: **{label_text}**")
            st.metric("نسبة التأكد", f"{confidence*100:.2f}%")
            
    except Exception as e:
        st.error(f"حدث خطأ غير متوقع: {e}")
        st.info("نصيحة: تأكد أن ملف labels.txt يحتوي على الأسماء صحيحة.")

else:
    st.warning("⚠️ النظام بانتظار ملفات keras_model.h5 و labels.txt")
