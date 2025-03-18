# 🧬 BCCD Detector - YOLOv10 Blood Cell Detection Dashboard

This Streamlit app allows you to detect and count **WBCs**, **RBCs**, and **Platelets** from blood smear images using a fine-tuned **YOLOv10** model.

---

## 🚀 Features

- 📤 Upload multiple blood cell images (JPG, PNG, JPEG formats)
- 🎯 Adjustable confidence threshold
- 📊 Detailed table and class-wise detection stats
- 📈 Compact bar chart visualization for class count
- 📥 Downloadable CSV detection report
- Clean & modern dark-themed UI
- Deployed on HuggingFace Space

---

## 🏃‍♀️ Run Locally

### Install Dependencies:

```bash
pip install -r requirements.txt
```

### Ensure YOLOv10 Fine-Tuned Model Exists:

Place your fine-tuned model `.pt` file inside the `models/` folder.

For example:

```bash
models/finetuned_yolov10s.pt
```

### Run the App:

```bash
streamlit run app.py
```

---

## 📂 Folder Structure

```
bccd-detector/
├── app.py                       # Streamlit application
├── models/
│   └── finetuned_yolov10s.pt    # Fine-tuned YOLOv10 model (you provide)
├── requirements.txt             # Project dependencies
└── README.md                    # Project overview (this file)
```

---

## 🌐 Deploy on Hugging Face Spaces

This app is **Hugging Face Spaces-ready**!

Deployed link: https://huggingface.co/spaces/shreyyy070/BCCD-Detection-Streamlit-App

## 📜 License

This project is for academic & demonstration purposes.  
For any use, please give proper credit.
