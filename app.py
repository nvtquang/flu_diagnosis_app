# app.py
import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "flu_pipeline.joblib"

@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)

def main():
    st.set_page_config(page_title="Chẩn đoán bệnh cảm cúm", layout="centered")
    st.title("🩺 Ứng dụng chẩn đoán bệnh cảm cúm (Naive Bayes)")

    # Tải model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        st.info("👉 Hãy chạy 'train_model.py' trước để tạo file flu_pipeline.joblib.")
        return

    st.header("Nhập triệu chứng của bạn")

    # Các triệu chứng (0/1)
    fever = st.checkbox("Sốt (fever)")
    cough = st.checkbox("Ho (cough)")
    fatigue = st.checkbox("Mệt mỏi (fatigue)")
    difficulty_breathing = st.checkbox("Khó thở (difficulty breathing)")

    # Giới tính
    gender = st.selectbox("Giới tính", ["Male", "Female", "Other"])

    # Nhóm tuổi (thay vì nhập tuổi)
    age_group = st.selectbox(
        "Nhóm tuổi",
        ["Trẻ em (<18)", "Thanh niên (18–35)", "Trung niên (36–60)", "Người già (>60)"]
    )

    # Map lại cho khớp model
    group_map = {
        "Trẻ em (<18)": "Tre_em",
        "Thanh niên (18–35)": "Thanh_nien",
        "Trung niên (36–60)": "Trung_nien",
        "Người già (>60)": "Nguoi_gia"
    }

    if st.button("🧠 Chẩn đoán"):
        input_data = {
            "fever": int(fever),
            "cough": int(cough),
            "fatigue": int(fatigue),
            "difficulty_breathing": int(difficulty_breathing),
            "gender": gender,
            "age_group": group_map[age_group]
        }

        input_df = pd.DataFrame([input_data])
        result = model.predict(input_df)[0]
        st.success(f"👉 Dự đoán: **{result}**")

        # Nếu có predict_proba thì hiển thị xác suất
        try:
            proba = model.predict_proba(input_df)[0]
            classes = model.classes_
            st.subheader("Xác suất từng loại bệnh:")
            for cls, p in zip(classes, proba):
                st.write(f"- {cls}: {p:.2f}")
        except Exception:
            pass

if __name__ == "__main__":
    main()
