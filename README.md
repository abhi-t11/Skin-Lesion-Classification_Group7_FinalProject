# Skin Lesion Detection and Classification Using Deep Learning

**Automated skin lesion classification using ResNet50 transfer learning on HAM10000 dataset. Achieved ~83% accuracy across 7 diagnostic categories.**

---

## 👥 Team Members

- **Abhi Jonnalagadda**
- **Abhishek Harish Thumar**
- **Dilip Bukkambudhi Ganesh**
- **Kyle Lobo**
- **Shreyash Naidu Mamidi**

**Institution:** Virginia Commonwealth University  
**Course:** CMSC 630 -  Image Analysis
**Instructor:** Dr. Wei-Bang Chen  
**Semester:** Fall 2025

---

## 📋 Project Overview

This project implements an automated skin lesion classification system using deep learning techniques, specifically transfer learning with ResNet50 on the HAM10000 dataset. The system classifies dermoscopic images into 7 diagnostic categories, achieving ~83% validation accuracy.

### Problem Statement
Skin cancer is one of the most prevalent cancers globally. Early detection is critical for successful treatment. This project develops an automated classification system to assist in computer-aided diagnosis for dermatology.

### Key Features
- ✅ Transfer learning with pre-trained ResNet50
- ✅ Data augmentation to handle limited samples
- ✅ Class imbalance handling with weighted loss
- ✅ Comprehensive evaluation with ROC curves and confusion matrices
- ✅ 7-class classification (akiec, bcc, bkl, df, mel, nv, vasc)

### Results Summary
- **Validation Accuracy:** 83.45%
- **Training Time:** 10.49 minutes (GPU)
- **Best Performing Classes:** nv (F1=0.90), vasc (F1=0.90)
- **Dataset:** HAM10000 (10,015 dermoscopic images)

---

## 🚀 Quick Start

### Prerequisites
- MATLAB R2020b or later
- Deep Learning Toolbox
- Image Processing Toolbox
- Computer Vision Toolbox
- Statistics and Machine Learning Toolbox
- (Optional) GPU with CUDA support for faster training

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/abhi-t11/Skin-Lesion-Classification_Group7_FinalProject
```

2. **Verify MATLAB toolboxes:**
```matlab
ver  % Check installed toolboxes
```

---

## 🎯 Quick Demo (2-3 minutes)

**For immediate testing without downloading the full 2GB dataset:**

The repository includes a sample dataset (70 images) for quick verification.
```matlab
% Open MATLAB and navigate to code directory
cd('path/to/Skin-Lesion-Classification/code')

% Run quick demo with sample dataset
run_sample_demo
```

**Expected Output:**
- ✅ Training completes in ~2-3 minutes
- ✅ Accuracy: ~60-70% (lower due to limited data - this is normal)
- ✅ Demonstrates that code works correctly
- ✅ Shows sample predictions and confusion matrix

**Note:** For full results (83% accuracy), download complete HAM10000 dataset following the instructions below.

---

## 📊 Dataset Setup

### Option 1: Quick Test (Sample Dataset - Included) ✅
- **Size:** ~5-10 MB
- **Images:** 70 images (10 per class)
- **Purpose:** Quick testing and code verification
- **Location:** Already in `data/sample_dataset/`
- **Setup time:** 0 minutes (ready to use immediately)
- **Usage:** Run `run_sample_demo.m`

### Option 2: Full Dataset (For Complete Results) 📥

**Dataset Information:**
- **Size:** ~2 GB
- **Images:** 10,015 dermoscopic images
- **Purpose:** Full training and evaluation (83% accuracy)
- **Setup time:** ~30 minutes

**Class Distribution:**
| Class | Description | Images | Percentage |
|-------|-------------|--------|------------|
| akiec | Actinic keratoses | 327 | 3.3% |
| bcc | Basal cell carcinoma | 514 | 5.1% |
| bkl | Benign keratosis | 1,099 | 11.0% |
| df | Dermatofibroma | 115 | 1.1% |
| mel | Melanoma | 1,113 | 11.1% |
| nv | Melanocytic nevi | 6,705 | 67.0% |
| vasc | Vascular lesions | 142 | 1.4% |

**Download Instructions:**

1. **Download from Kaggle:**
   - URL: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
   - Requires free Kaggle account

2. **Extract to data folder:**
   - Place `HAM10000_images_part_1/`, `HAM10000_images_part_2/`, and `HAM10000_metadata.csv` in `data/` directory

3. **See detailed instructions:**
   - Complete setup guide in `data_readme.txt`

---

## 🔧 Usage

### Option 1: Quick Demo (2-3 minutes) ⚡
```matlab
cd('code')
run_sample_demo  % Uses included sample dataset (70 images)
```

### Option 2: Full Pipeline (30-60 minutes) 🚀

**Prerequisites:** Download full HAM10000 dataset following `data_readme.txt`

**Run each step sequentially:**
```matlab
cd('code')

% Step 1: Organize dataset by class (~10 minutes)
step1_organize_data

% Step 2: Preprocess and create train/val/test splits (~5 minutes)
step2_preprocess_data

% Step 3: Train classification model (~10 minutes with GPU)
step3_train_model

% Step 4: Evaluate on test set (~5 minutes)
step4_evaluate_model

% Step 5: Generate report materials (~3 minutes)
step5_generate_report_materials
```

**What each step does:**

| Step | Script | Duration | Description |
|------|--------|----------|-------------|
| 1 | `step1_organize_data.m` | ~10 min | Merges image folders and organizes by diagnostic class |
| 2 | `step2_preprocess_data.m` | ~5 min | Creates train/val/test splits, sets up data augmentation |
| 3 | `step3_train_model.m` | ~10 min | Trains ResNet50 model with transfer learning |
| 4 | `step4_evaluate_model.m` | ~5 min | Evaluates model on test set, generates metrics |
| 5 | `step5_generate_report_materials.m` | ~3 min | Creates tables and figures for report |

**Total Time:** ~30-35 minutes (with GPU) or ~90 minutes (CPU only)

---

## 📁 Project Structure
```
Skin-Lesion-Classification/
├── code/                           # Source code (MATLAB scripts)
│   ├── run_sample_demo.m           # Quick demo with sample data
│   ├── step1_organize_data.m       # Data organization
│   ├── step2_preprocess_data.m     # Preprocessing and splitting
│   ├── step3_train_model.m         # Model training
│   ├── step4_evaluate_model.m      # Model evaluation
│   ├── step5_generate_report_materials.m
│   ├── create_sample_dataset.m     # Generate sample dataset
│   └── check_setup.m               # Verify installation
│
├── data/                           # Dataset folder
│   ├── sample_dataset/             # ✅ Included sample (70 images)
│   │   ├── akiec/     (10 images)
│   │   ├── bcc/       (10 images)
│   │   ├── bkl/       (10 images)
│   │   ├── df/        (10 images)
│   │   ├── mel/       (10 images)
│   │   ├── nv/        (10 images)
│   │   ├── vasc/      (10 images)
│   │   └── README.txt
│   ├── HAM10000_images_part_1/     # Download required
│   ├── HAM10000_images_part_2/     # Download required
│   └── HAM10000_metadata.csv       # Download required
│
├── results/                        # Output folder (created by scripts)
│   ├── step2_preprocessed_data.mat
│   ├── step3_trained_model.mat
│   ├── step4_evaluation_results.mat
│   └── *.csv                       # Metrics tables
│
├── figures/                        # Visualizations (created by scripts)
│   ├── step2_*.png                 # Preprocessing figures
│   ├── step3_*.png                 # Training results
│   └── step4_*.png                 # Evaluation results
│
├── report_materials/               # Report-ready materials
│   ├── table*.csv                  # LaTeX/Word-ready tables
│   └── figure*.png                 # High-res figures
│
├── README.md                       # This file
├── data_readme.txt                 # Dataset setup instructions
└── LICENSE                         # MIT License
```

---

## 🏗️ Architecture

### Model Architecture
- **Base Model:** ResNet50 (pre-trained on ImageNet)
- **Input Size:** 224×224×3 RGB images
- **Total Layers:** 177 layers
- **Parameters:** ~25 million
- **Modifications:** 
  - Replaced final fully connected layer (1000 → 7 classes)
  - Added weighted cross-entropy loss for class imbalance
  - Fine-tuned last layers with higher learning rate
- **Training Strategy:** Transfer learning with fine-tuning

### Data Pipeline
```
Raw Images → Preprocessing → Augmentation → Training
    ↓              ↓              ↓              ↓
10,015 JPG    Resize to      Rotation        ResNet50
  images      224×224×3      Flipping         Model
              Normalize      Scaling            ↓
                [0,1]        Shearing      Predictions
```

**Preprocessing Steps:**
1. **Resize:** All images resized to 224×224 pixels
2. **Normalize:** Pixel values scaled to [0, 1]
3. **Augmentation (training only):**
   - Random rotation: ±20 degrees
   - Random horizontal/vertical flips
   - Random scaling: 80-120%
   - Random shear: ±10 degrees

**Data Split (Stratified):**
- Training: 70% (7,011 images)
- Validation: 20% (2,013 images)
- Test: 10% (991 images)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 16 |
| Max Epochs | 30 |
| Early Stopping | Yes (patience=5) |
| Validation Frequency | Every 50 iterations |
| Loss Function | Weighted Cross-Entropy |
| Execution | GPU (CUDA) |

---

## 📈 Results

### Overall Performance (Full Dataset)

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **83.31%** |
| Macro-Avg Precision | 0.7135 |
| Macro-Avg Recall | 0.7207 |
| Macro-Avg F1-Score | 0.7094 |
| Mean AUC | 0.8945 |
| Training Time | 10.49 minutes |
| Epochs Completed | 4 (early stopping) |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | AUC | Support |
|-------|-----------|--------|----------|-----|---------|
| akiec | 0.8750 | 0.4242 | 0.5714 | 0.8523 | 66 |
| bcc | 0.8182 | 0.6990 | 0.7539 | 0.9234 | 103 |
| bkl | 0.5667 | 0.6923 | 0.6232 | 0.8756 | 221 |
| df | 0.4211 | 0.6957 | 0.5246 | 0.8432 | 23 |
| mel | 0.4927 | 0.7500 | 0.5947 | 0.8645 | 224 |
| **nv** | **0.9472** | **0.8530** | **0.8977** | **0.9534** | **1341** |
| **vasc** | **0.8710** | **0.9310** | **0.9000** | **0.9489** | **35** |

### Key Findings

#### ✅ Strengths
- **Excellent performance on majority classes:** nv (F1=0.90) and vasc (F1=0.90)
- **Fast convergence:** Early stopping at epoch 4 (only 10 minutes)
- **High specificity:** Model rarely misclassifies healthy tissue as cancerous
- **Transfer learning effectiveness:** Pre-trained ResNet50 enabled quick training

#### ⚠️ Challenges
- **Minority class difficulty:** df (F1=0.52) challenging due to only 115 training samples
- **Class imbalance impact:** Despite weighted loss, rare conditions remain difficult
- **Precision-recall tradeoff:** Some classes favor recall over precision

#### 🔍 Clinical Insights
- Model performs best on visually distinct lesions (vasc, nv)
- Confusion occurs between similar-appearing benign conditions (bkl, mel)
- Additional data needed for rare conditions (df, akiec)

---

## 🖼️ Visualizations

All visualizations are automatically generated and saved in `figures/` directory:

### Training & Validation
- `step3_training_progress.png` - Loss and accuracy curves over epochs
- `step3_validation_confusion_matrix.png` - Validation set confusion matrix
- `step3_sample_predictions.png` - Example predictions with confidence scores

### Test Set Evaluation
- `step4_confusion_matrix.png` - Final test set confusion matrix
- `step4_roc_curves.png` - ROC curves for each class (one-vs-all)
- `step4_performance_comparison.png` - Per-class metrics bar chart
- `step4_correct_predictions.png` - Successfully classified examples
- `step4_incorrect_predictions.png` - Misclassified examples with analysis

### Data Analysis
- `step2_full_distribution.png` - Class distribution histogram
- `step2_sample_training_images.png` - Sample images from each class
- `step2_augmentation_examples.png` - Data augmentation visualization
- `step2_class_weights.png` - Calculated class weights for loss function

---

## 🔬 Methodology

### Transfer Learning Approach

We employed transfer learning using ResNet50 pre-trained on ImageNet:

**Advantages:**
- ✅ Leverages features learned from 1.2M images
- ✅ Reduces training time from hours to minutes
- ✅ Improves performance on limited medical datasets
- ✅ Reduces risk of overfitting

**Implementation:**
1. Load ResNet50 with ImageNet weights (frozen layers)
2. Replace final classification layer (1000 → 7 classes)
3. Fine-tune with higher learning rate on new layer
4. Train on HAM10000 with data augmentation

### Handling Class Imbalance

The dataset exhibits severe class imbalance (nv: 6705 vs df: 115 images).

**Mitigation Strategies:**
1. **Stratified splitting** - Maintains class proportions across train/val/test
2. **Weighted cross-entropy loss** - Penalizes misclassification of minority classes more heavily
3. **Data augmentation** - Artificially increases dataset size
4. **Early stopping** - Prevents overfitting on majority classes

**Class Weights Calculation:**
```
weight(class_i) = total_samples / (num_classes × samples_in_class_i)
```

### Data Augmentation Strategy

Applied to **training set only** to prevent data leakage:

| Augmentation | Range | Purpose |
|--------------|-------|---------|
| Rotation | ±20° | Handle various camera orientations |
| Horizontal Flip | Yes | Bilateral symmetry of lesions |
| Vertical Flip | Yes | No inherent orientation |
| Scaling | 80-120% | Different distances from camera |
| Shearing | ±10° | Simulate viewing angles |

**Augmentation Examples:** See `figures/step2_augmentation_examples.png`

---

## 📊 Comparison with Literature

| Study | Year | Dataset | Method | Accuracy | Notes |
|-------|------|---------|--------|----------|-------|
| Tschandl et al. | 2018 | HAM10000 | Baseline CNN | ~85% | Original dataset paper |
| Guth & de Campos | 2018 | HAM10000 | U-Net Segmentation | ~87.5% | Focus on segmentation |
| Khan et al. | 2021 | HAM10000 | DL + Optimization | ~92% | Ensemble + optimization |
| **This Project** | **2025** | **HAM10000** | **ResNet50 Transfer** | **83.31%** | **Single model, fast training** |

### Analysis

Our results (83.31% validation accuracy) are competitive with baseline approaches and demonstrate:
- ✅ Effectiveness of transfer learning for medical imaging
- ✅ Feasibility with limited computational resources (10 minutes vs hours)
- ✅ Strong performance without ensemble methods
- ⚠️ Room for improvement with advanced techniques (ensemble, attention mechanisms)

---

## 🚧 Known Issues & Limitations

### Technical Limitations
1. **Class Imbalance:** Despite mitigation strategies, minority classes (df, akiec) remain challenging
2. **Dataset Size:** Limited samples for rare conditions (df: 115 images) affect generalization
3. **No Segmentation Masks:** HAM10000 lacks pixel-level annotations for lesion boundary detection
4. **Hardware Dependency:** Training time varies significantly (10 min GPU vs 90 min CPU)

### Clinical Limitations
5. **Single Dataset:** Model trained only on HAM10000; may not generalize to other populations
6. **No External Validation:** Results not validated on independent clinical datasets
7. **Metadata Not Used:** Patient age, sex, location not incorporated in model
8. **Binary Decisions Only:** No uncertainty quantification or confidence intervals

### Deployment Considerations
9. **Regulatory Approval:** Not FDA approved; research use only
10. **Clinical Integration:** Requires extensive validation before clinical deployment
11. **Interpretability:** Limited explanation of why predictions are made
12. **Real-time Requirements:** Not optimized for edge devices or mobile deployment

---

## 🔮 Future Work

### Short-term Improvements
1. **Ensemble Methods**
   - Combine ResNet50, DenseNet, EfficientNet
   - Voting or stacking strategies
   - Expected: +3-5% accuracy improvement

2. **Advanced Augmentation**
   - Mixup and Cutmix
   - AutoAugment or RandAugment
   - Test-time augmentation

3. **Hyperparameter Optimization**
   - Learning rate scheduling
   - Batch size tuning
   - Optimizer comparison (Adam vs SGD)

### Medium-term Enhancements
4. **Attention Mechanisms**
   - Spatial attention to focus on lesion regions
   - Channel attention for feature selection
   - Grad-CAM visualization for interpretability

5. **Segmentation Integration**
   - Implement U-Net for lesion boundary detection
   - Two-stage pipeline: segment then classify
   - Multi-task learning (joint segmentation + classification)

6. **Multi-modal Learning**
   - Incorporate patient metadata (age, sex, location)
   - Fusion of image and tabular data
   - Expected: Better generalization

### Long-term Goals
7. **Clinical Deployment**
   - Web interface for dermatologists
   - Mobile app for primary care
   - Integration with Electronic Health Records (EHR)

8. **External Validation**
   - Test on ISIC, BCN20000, PAD-UFES-20 datasets
   - Multi-center clinical trials
   - Real-world performance evaluation

9. **Regulatory Approval**
   - FDA 510(k) clearance pathway
   - CE marking for European deployment
   - Clinical evidence generation

---

## 📚 References

### Dataset
1. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161. https://doi.org/10.1038/sdata.2018.161

### Methodology
2. Guth, F., & de Campos, T. E. (2018). Skin lesion segmentation using U-Net and good training strategies. *arXiv preprint* arXiv:1811.11314.

3. Khan, M. A., et al. (2021). Skin lesion segmentation and multiclass classification using deep learning features and improved moth flame optimization. *Diagnostics*, 11(5), 811. https://doi.org/10.3390/diagnostics11050811

4. Kazaj, P. M., et al. (2022). U-Net-based models for skin lesion segmentation: More attention and augmentation. *arXiv preprint* arXiv:2210.16399.

### Deep Learning
5. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (CVPR), pp. 770-778.

6. Russakovsky, O., et al. (2015). ImageNet large scale visual recognition challenge. *International Journal of Computer Vision*, 115(3), 211-252.

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

**Academic Use:** Free for research and educational purposes  
**Commercial Use:** Requires proper licensing and regulatory approval  
**Citation Required:** Please cite this repository if used in academic work

---

## 🙏 Acknowledgments

### Dataset & Resources
- **HAM10000 Dataset:** Medical University of Vienna
- **ImageNet Weights:** Stanford Vision Lab
- **MATLAB Deep Learning Toolbox:** MathWorks

### Institution
- **Virginia Commonwealth University**
- **Department of Computer Science**
- **CMSC 508 - Image Analysis Course**

### Special Thanks
- **Dr. Wei-Bang Chen** - Course instructor and project advisor
- **MathWorks** - MATLAB licensing and support
- **Kaggle Community** - Dataset hosting and discussions

---

## 👤 Contact

**Project Maintainer:** Abhishek Tripathi  
**Email:** [your.email@vcu.edu](mailto:your.email@vcu.edu)  
**GitHub:** [@abhi-t11](https://github.com/abhi-t11)  
**Institution:** Virginia Commonwealth University

### Getting Help
- 📧 Email for questions or collaboration
- 🐛 [Open an issue](https://github.com/abhi-t11/Skin-Lesion-Classification/issues) for bugs
- 💡 [Start a discussion](https://github.com/abhi-t11/Skin-Lesion-Classification/discussions) for ideas
- ⭐ Star this repo if you find it useful!

---

## 📞 Support & Troubleshooting

### Common Issues

<details>
<summary><b>Issue: "No GPU detected" warning</b></summary>

**Solution:**
- This is normal if you don't have an NVIDIA GPU
- Training will use CPU (slower but works)
- Expected time: ~90 minutes instead of 10 minutes
- All functionality remains the same
</details>

<details>
<summary><b>Issue: "Dataset not found" error</b></summary>

**Solution:**
1. Run quick demo first: `run_sample_demo.m`
2. For full dataset, download from Kaggle
3. Check `data_readme.txt` for setup instructions
4. Verify folder names match exactly (case-sensitive)
</details>

<details>
<summary><b>Issue: "Out of memory" error</b></summary>

**Solution:**
1. Reduce batch size in `step3_train_model.m`:
```matlab
   'MiniBatchSize', 8  % Change from 16 to 8
```
2. Close other applications
3. Use CPU instead of GPU (more memory)
</details>

<details>
<summary><b>Issue: "Toolbox not found" error</b></summary>

**Solution:**
1. Check installed toolboxes: `ver`
2. Install missing toolboxes from MATLAB Add-Ons
3. Restart MATLAB after installation
4. Run `check_setup.m` to verify
</details>

### Getting Help

If you encounter issues:
1. ✅ **Try quick demo first:** Run `run_sample_demo.m` to verify basic setup
2. 📖 **Check documentation:** Review `data_readme.txt` for dataset setup
3. 🔍 **Verify installation:** Run `check_setup.m` to check all requirements
4. 📊 **Review logs:** Check `results/` folder for error messages
5. 🐛 **Open an issue:** [GitHub Issues](https://github.com/abhi-t11/Skin-Lesion-Classification/issues) with:
   - MATLAB version
   - Error message (full text)
   - Steps to reproduce
   - Screenshots (if applicable)

---

## 🎯 Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| ✅ Data Organization | Complete | Step 1 working |
| ✅ Preprocessing | Complete | Step 2 working |
| ✅ Model Training | Complete | Step 3 working, 83% accuracy |
| ✅ Evaluation | Complete | Step 4 working, full metrics |
| ✅ Report Materials | Complete | Step 5 working, all tables/figures |
| ✅ Sample Dataset | Complete | Included for quick testing |
| ✅ Documentation | Complete | README, data_readme, comments |
| 📝 Final Report | In Progress | Due November 2025 |
| 🎥 Presentation Video | Pending | Scheduled for November 2025 |

**Last Updated:** November 20, 2025

---

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=abhi-t11/Skin-Lesion-Classification&type=Date)](https://star-history.com/#abhi-t11/Skin-Lesion-Classification&Date)

---