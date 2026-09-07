# Industrial Surface Defect Classifier (Edge Vision) 🏭👁️

> **An automated industrial computer vision inspection system using Deep Convolutional Neural Networks (MobileNetV2 Transfer Learning) to detect and classify defects on hot-rolled steel surfaces by Abhi Gurjar.**

---

## 💡 What Is This Project? (Explained Like a Human)

In modern manufacturing plants—especially steel mills, automotive stamping lines, and aerospace fabrication—metal sheets roll through high-speed conveyors at up to 30 km/h (18 mph).

If a microscopic crack, foreign particle, or mill scale defect gets stamped into that steel and goes unnoticed:
- A car manufacturer might build an entire vehicle door that snaps under stress.
- A pressurized pipe or bridge beam could fail catastrophically.
- The factory faces millions of dollars in product recalls and warranty liability.

Traditionally, factories relied on human inspectors standing beside conveyor belts. But human eyes get tired after 20 minutes, and no human can inspect thousands of feet of steel moving at highway speeds with 100% consistency.

**This project replaces error-prone manual inspection with Computer Vision.** By combining Deep Learning Transfer Learning (`MobileNetV2`) with high-frequency surface texture analysis, the system detects, classifies, and flags defective steel in milliseconds before coils leave the factory floor.

---

## 🔍 The 6 Defect Categories Detected

| Defect Type | Manufacturing Cause | Failure Risk |
|---|---|---|
| **Crazing** | Network of fine micro-cracks from thermal fatigue | Structural fatigue under tension |
| **Inclusion** | Non-metallic chemical particles pressed into metal | Weakened localized tensile strength |
| **Patches** | Irregular oxidized scale or localized surface peeling | Uneven coating & paint adhesion failure |
| **Pitted Surface** | Pinhole cavities caused by acid pickling or corrosion | Accelerated environmental rusting |
| **Rolled-in Scale** | Hard iron oxide flakes embedded during rolling | Surface roughness & cracking during forming |
| **Scratches** | Sharp mechanical abrasive grooves from friction | Stress concentration points prone to tear |

---

## 🏛️ Model Architecture & Transfer Learning

Instead of training a billion-parameter network from scratch—which requires months of computing and massive server farms—we apply **Transfer Learning**:
1. **Pretrained Backbone**: Leverages `MobileNetV2` pretrained on ImageNet to extract general edge, gradient, and texture features.
2. **Custom Industrial Head**: Freezes earlier generic layers and trains a custom classification head consisting of Global Average Pooling, Dropout (0.3) for regularization, and a Softmax output layer across the 6 defect categories.
3. **Edge Optimization**: Compact parameter footprint (<15 MB) enables real-time deployment on edge devices like **NVIDIA Jetson**, **Raspberry Pi 5**, or industrial inspection cameras with sub-15ms inference latency.

---

## 🚀 How to Run in 5 Seconds

The repository includes sample industrial steel defect images and a standalone inference runner:

```bash
# 1. Clone the repository
git clone https://github.com/abhigurjar101/industrial-surface-defect-detection.git
cd industrial-surface-defect-detection

# 2. Run the edge inference runner
python3 inference.py
```

### What You Will See:
- Rapid classification of all 6 sample defect types.
- Calculated texture roughness, surface mean intensity, and confidence percentages.
- Automated MES (Manufacturing Execution System) severity assessment (`CRITICAL` vs `MEDIUM`).

---

## 📁 Repository Structure

```
industrial-surface-defect-detection/
├── notebooks/
│   └── Transfer_Learning_Surface_Defect_AbhiGurjar.ipynb  # End-to-end training notebook
├── samples/                                               # Sample industrial test images
│   ├── crazing_sample.jpg
│   ├── inclusion_sample.jpg
│   ├── patches_sample.jpg
│   ├── pitted_surface_sample.jpg
│   ├── rolled-in_scale_sample.jpg
│   └── scratches_sample.jpg
├── inference.py                                           # Real-time edge inference engine
├── requirements.txt                                       # Torch, Pillow, Scikit-Learn
└── README.md                                              # You are here! Simple human documentation
```

---

## 👤 Author

**Abhi Gurjar**  
- Portfolio: [abhigurjar.vercel.app](https://abhigurjar.vercel.app)  
- GitHub: [@abhigurjar101](https://github.com/abhigurjar101)  
- LinkedIn: [Abhi Gurjar](https://in.linkedin.com/in/abhi-gurjar-b13067203)
