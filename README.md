# 🦴 MediScan AI: Bone Fracture Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green)

A lightweight, AI-powered web application that analyzes X-ray images to automatically detect and localize bone fractures. Built with **YOLOv8**, **Streamlit**, and **OpenCV**, this tool provides real-time inference with a clean, medical-grade interface.

---

## 🚀 Live Demo
https://bone-fracture-detections.streamlit.app/
---

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/41ee82dc-245a-4bec-9917-91e9ad49e21f" />


## 🎯 Key Features

* **Real-Time Detection:** Instantly analyzes uploaded X-ray images.
* **Visual Localization:** Draws precise red bounding boxes around detected fractures.
* **Dual Mode:**
    * **Demo Mode:** Uses standard YOLOv8n (detects objects like "vase" or "tie" as placeholders for testing).
    * **Medical Mode:** Upload your own trained `.pt` file for actual fracture detection.
* **Privacy Focused:** No images are saved; processing happens in-memory.
* **Responsive UI:** Dark-themed, medical-style interface powered by Streamlit.

---

## 🛠️ Tech Stack

* **Framework:** [Streamlit](https://streamlit.io/)
* **Computer Vision:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* **Image Processing:** OpenCV & PIL
* **Language:** Python 3.10+

---

## 📦 Installation (Local)

Follow these steps to run the app on your own machine:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/bone-fracture-detection.git](https://github.com/your-username/bone-fracture-detection.git)
    cd bone-fracture-detection
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

---

## ☁️ Deployment (Streamlit Cloud)

If deploying to Streamlit Cloud, ensure your repository contains:
1.  `app.py` (Main application code)
2.  `requirements.txt` (Python libraries)
3.  `packages.txt` (System dependencies like `libgl1`)

**Configuration for `packages.txt`:**
```text
libgl1
