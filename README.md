# Thermal-runaway-detection-unet

# 📌 Overview
This project presents a deep learning–based thermal management system for early detection of thermal runaway in electric vehicle (EV) battery packs using infrared thermal imaging and U-Net semantic segmentation.

# 🚀 Key Contributions

- Pixel-level hotspot region detection using U-Net 
- Real-time inference (<100 ms latency)
- Active cooling and isolation of fault cell using MOSFET-based cirucuit breakers
- 25% reduction in thermal runaway incidents(experimental)

# 🧠 Model Architecture

- U-Net (Encoder–Decoder and bottkleneck with skip connections)
- Loss  :  Focal(IoU) + Dice Loss
- Input : 128×128 thermal images

# 📊 Results

| Metric           | Value  |
| Dice Coefficient | 0.84   |
| IoU              | 0.67   |
| Accuracy         | 85.3%  |
| Inference Time   |<100 ms |

# 🛠 Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- Raspberry Pi 3
- MATLAB (simulation)

# 📁 Repository Structure

thermal-runaway-detection-unet/
│
├── README.md
│
├── docs/
│   └── Project_Report.pdf
│
├── notebooks/
│   └── UNet_Thermal_Runaway_Training.ipynb
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── utils.py
│
├── dataset/
│   ├── images/     
│   └── masks/
│
├── results/
│   ├── predictions/
│   └── metrics.png
│
├── requirements.txt
│
└── .gitignore


# ▶️ How to Run
```
pip install -r requirements.txt
python src/train.py
python src/inference.py
