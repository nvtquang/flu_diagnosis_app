import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
CSV_PATH = "flu_symptoms_dataset.csv"
MODEL_PATH = "flu_pipeline.joblib"

# 🧩 Hàm chuyển tuổi thành nhóm
def chuyen_nhom_tuoi(age):
    if age < 18:
        return "Tre_em"
    elif age <= 35:
        return "Thanh_nien"
    elif age <= 60:
        return "Trung_nien"
    else:
        return "Nguoi_gia"

def train():
    # Đọc dữ liệu
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Kiểm tra các cột chính
    print("Các cột có trong dữ liệu:", df.columns.tolist())

    # Chuyển Yes/No thành 1/0
    yes_no_map = {"Yes" : 1, "No" : 0}
    for col in ["fever","cough","fatigue","difficulty_breathing"]:
        df[col] = df[col].map(yes_no_map)

    # Tạo nhóm tuổi
    if "age" not in df.columns:
        raise ValueError("Không tìm thấy cột 'age' trong dữ liệu!")
    df["age_group"] = df["age"].apply(chuyen_nhom_tuoi)

    # Xác định các cột đặc trưng
    symptom_cols = ["fever", "cough", "fatigue", "difficulty_breathing"]
    categorical = ["gender", "age_group"]
    label_col = "outcome_variable"

    for col in symptom_cols + categorical + [label_col]:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột {col} trong dữ liệu!")

    X = df[symptom_cols + categorical]
    y = df[label_col].astype(str)

    # Tiền xử lý: OneHotEncode cho các biến phân loại
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        ],
        remainder="passthrough"
    )

    # Tạo pipeline: tiền xử lý + Naive Bayes
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", GaussianNB())
    ])

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Huấn luyện
    model.fit(X_train, y_train)

    # Đánh giá
    y_pred = model.predict(X_test)
    print("Độ chính xác:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Đã lưu mô hình vào file: {MODEL_PATH}")

if __name__ == "__main__":
    train()