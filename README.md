# Localization Using Darknet and YOLOv3

This project implements **real-time object detection and localization** using **YOLOv3** with a **Darknet backbone**. It’s designed for detecting multiple objects in images efficiently, making it suitable for experimentation, learning, or integration into larger systems.

---

## Features

- Real-time object detection using YOLOv3  
- Multi-scale detection for accurate localization  
- Lightweight Darknet backbone for faster inference  

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/custom-darknet-yolov3.git
cd custom-darknet-yolov3
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 4. Preparing YOLO Weights

This project uses the YOLOv3 weights file: (put it in weights/)

[yolov3.weights](https://www.kaggle.com/datasets/aruchomu/data-for-yolo-v3-kernel?select=yolov3.weights)

### 5. Running the Detection

1. Add your images to the input/ folder.

2. Run the detection script:
```bash
python detect.py
```

### 6. Test on Images

![dog](https://github.com/user-attachments/assets/e1ae747d-6a20-49a1-827f-490f6b0c2f6d)

![det_dog](https://github.com/user-attachments/assets/a659d41a-53d9-4f94-812b-f30676a184e5)


## References

1. **YOLOv1 Paper:** [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)  
2. **YOLOv2 Paper:** [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)  
3. **YOLOv3 Paper:** [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)  
4. **YOLOv3 Weights Dataset (Kaggle):** [yolov3.weights](https://www.kaggle.com/datasets/aruchomu/data-for-yolo-v3-kernel?select=yolov3.weights)
