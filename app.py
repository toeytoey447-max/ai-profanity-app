import streamlit as st
import pickle
from pythainlp.tokenize import word_tokenize

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def preprocess(text):
    return " ".join(word_tokenize(text, engine="newmm"))

st.title("🧠 AI ตรวจจับคำหยาบในแชต")

text = st.text_area("พิมพ์ข้อความที่ต้องการตรวจสอบ")

if st.button("ตรวจสอบ"):
    t = preprocess(text)
    x = vectorizer.transform([t])
    result = model.predict(x)[0]

    if result == 1:
        st.error("🔴 พบคำหยาบ")
    else:
        st.success("🟢 ข้อความสุภาพ")
