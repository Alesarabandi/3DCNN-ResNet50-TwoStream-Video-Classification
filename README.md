# 🎬 Temporal Action Recognition in Motion on the HMDB51 Dataset

## Project Overview
This project explores various deep learning architectures for **Temporal Action Recognition** on the **HMDB51 dataset**, a widely used benchmark for human motion analysis. The goal was to accurately classify singular actions in short video clips by analyzing both **spatial appearance (RGB frames)** and **temporal motion (Optical Flow)**.

The final system is a **Two-Stream Convolutional Network** utilizing late fusion, which effectively combines the strengths of dedicated spatial and temporal streams to achieve robust classification performance.

---

## 🎯 Key Objectives
* Perform comprehensive **Exploratory Data Analysis (EDA)** on the HMDB51 dataset.
* Implement efficient techniques for **RGB Frame** and **Optical Flow** extraction.
* Develop and compare multiple single-stream deep learning models (2D CNN, 3D CNN) to establish a performance baseline.
* Construct and train a **Two-Stream CNN** using a **Late Fusion** strategy.
* Analyze the contribution of the spatial and temporal streams to the final classification accuracy.

---

## 🛠️ Project Stages and Models

The project was executed in a sequential, iterative manner, documented across seven notebooks and a final presentation.

### **Phase 1: Data Preparation & Exploration**

| File | Description | Key Result |
| :--- | :--- | :--- |
| `1.EDA` | **Exploratory Data Analysis** | Initial setup, unzipping the HMDB51 dataset, and confirming video duration distribution across all 51 classes. |
| `2.frames_extraction` | **RGB Frame Extraction** | Script for preprocessing the raw videos by extracting sequential RGB frames, which serve as the input for the spatial stream. |
| `6.flow_extraction` | **Optical Flow Extraction** | Implementation of an algorithm (e.g., Farneback or TV-L1) to compute **dense Optical Flow** fields, which explicitly capture pixel-level motion for the temporal stream. |

### **Phase 2: Single-Stream Model Development**

This phase involved building and testing single-stream architectures to understand their independent performance.

#### **1. Pure Temporal Models (3D CNN)**
| File | Model | Key Features & Outcome |
| :--- | :--- | :--- |
| `3.3DCNN_v1.0` | **Initial 3D CNN** | A foundational 3D Convolutional Network model designed to learn spatiotemporal features directly from video clips (stacked frames). |
| `4.3DCNN-v2.0` | **Refined 3D CNN** | An optimized, smaller architecture with enhanced use of **Dropout** and refined hyperparameters. This version was significantly **more efficient and achieved superior accuracy** compared to `v1.0`, highlighting the importance of smart architecture design over parameter count. |

#### **2. Pure Spatial Model (2D CNN for Spatial Stream)**
| File | Model | Key Features & Outcome |
| :--- | :--- | :--- |
| `5.ResNet50_finetuned` | **Fine-tuned ResNet-50** | Utilized a powerful pre-trained **ResNet-50** model, fine-tuned on the single RGB frames, to serve as the **Spatial Stream** in the final Two-Stream architecture. |

### **Phase 3: Final Architecture & Fusion**

| File | Model | Key Features & Outcome |
| :--- | :--- | :--- |
| `7.TwoStreams` | **Two-Stream Network** | The final architecture combining the Spatial (ResNet-50 on RGB) and Temporal (3D CNN on Optical Flow) streams using **Late Fusion**. This allows the model to leverage both appearance and explicit motion signals for classification. |

---

## 📊 Key Results and Insights

The Two-Stream approach, based on the original work by Simonyan and Zisserman, was implemented with the following key findings:

| Model | Top-1 Accuracy (Example) | Insight |
| :--- | :--- | :--- |
| **Spatial Stream (RGB)** | (e.g., 61.2%) | The appearance-based stream was the **dominant performer**, successfully recognizing actions largely based on context and static features. |
| **Two-Stream (Late Fusion)** | (e.g., 62.1%) | The fusion provided a **marginal, but positive, increase in Top-1 accuracy** over the dominant RGB stream alone, validating the hypothesis that explicit motion information still contributes to classification. |

---

## 💻 Setup and Dependencies

To run this project, you will need the following key libraries and tools.

1.  **Programming Language:** Python
2.  **Core Libraries:**
    * `tensorflow`/`keras`
    * `numpy`
    * `opencv-python` (for frame and optical flow extraction)
    * `scikit-learn`
3.  **Data:** The **HMDB51** dataset. *(Note: Due to its size, the HMDB51 dataset is not included in this repository and must be downloaded separately.)*

```bash
# Example setup
# Install dependencies
pip install tensorflow opencv-python numpy scikit-learn
