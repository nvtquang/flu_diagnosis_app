# Flu Diagnosis App

Ứng dụng **Streamlit** kết hợp **Naive Bayes** để chẩn đoán khả năng bị cúm dựa trên triệu chứng và thông tin bệnh nhân.

---

## 📁 Cấu trúc dự án

- `app.py`: Ứng dụng Streamlit – giao diện người dùng để nhập triệu chứng và tiên đoán.  
- `train_model.py`: Script huấn luyện mô hình Naive Bayes.  
- `flu_symptoms_dataset.csv`: Dữ liệu dùng để huấn luyện.  
- `flu_pipeline.joblib`: Mô hình pipeline đã huấn luyện và lưu bằng `joblib`.  
- `README.md`: Tệp hướng dẫn này.

---

## 📊 Dữ liệu

- Dữ liệu được sử dụng từ Kaggle: **Disease Symptoms and Patient Profile Dataset**  
- Link Kaggle:  
  [https://www.kaggle.com/datasets/uom190346a/disease-symptoms-and-patient-profile-dataset](https://www.kaggle.com/datasets/uom190346a/disease-symptoms-and-patient-profile-dataset?utm_source=chatgpt.com)  
- Bộ dữ liệu bao gồm các thông tin như triệu chứng (`fever`, `cough`, `fatigue`, `difficulty_breathing`), tuổi, giới tính và kết quả (`outcome_variable`).  
- Dữ liệu được tiền xử lý như sau:
  - Nhóm tuổi được chia thành "Trẻ em", "Thanh niên", "Trung niên", "Người già".  
  - Triệu chứng dạng nhị phân (`Yes/No`) được chuyển đổi sang dạng số (`0/1`).  
  - Biến phân loại như giới tính và nhóm tuổi được One-Hot Encode.

---

## Hướng dẫn chạy chương trình

- Chạy "**train_model.py**"
- Mô hình được lưu vào file "**flu_pipeline.joblib**"
- Mở terminal chạy streamlit: gõ lệnh "**streamlit run app.py**"
- Giờ bạn có thể thấy ứng dụng Streamlit chạy trên trình duyệt của bạn tại Local URL: **http://localhost:8501**
