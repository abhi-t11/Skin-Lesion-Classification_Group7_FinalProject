HAM10000 Dataset Setup Instructions
====================================

The HAM10000 dataset is NOT included in this repository due to its large size (~2GB).
You must download it separately from Kaggle.

Download Instructions:
----------------------

Source: Skin Cancer MNIST: HAM10000 (Kaggle)
URL: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Steps:
1. Create a free Kaggle account (if you don't have one)
2. Go to: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
3. Click "Download" button (downloads as archive.zip or skin-cancer-mnist-ham10000.zip)
4. Extract the downloaded ZIP file

Files Included in Download:
---------------------------
After extraction, you will get:
- HAM10000_images_part_1/ folder (contains ~5000 .jpg files)
- HAM10000_images_part_2/ folder (contains ~5000 .jpg files)
- HAM10000_metadata.csv (contains labels and patient information)
- hmnist_8_8_L.csv (8x8 grayscale - not used in this project)
- hmnist_8_8_RGB.csv (8x8 RGB - not used in this project)
- hmnist_28_28_L.csv (28x28 grayscale - not used in this project)
- hmnist_28_28_RGB.csv (28x28 RGB - not used in this project)

Note: The hmnist CSV files are downsampled versions (MNIST-style) 
      and are NOT used in this project. We use the full-resolution
      JPG images from the two image folders.

Installation:
-------------

1. Download and extract the dataset from Kaggle (see above)

2. Place the extracted files in your project's data/ folder:

   Your project structure should look like:
   
   Skin Disease Detection/
   └── data/
       ├── HAM10000_images_part_1/       (folder with ~5000 images)
       ├── HAM10000_images_part_2/       (folder with ~5000 images)
       ├── HAM10000_metadata.csv         (CSV file, 551 KB)
       ├── hmnist_8_8_L.csv              (optional, not used)
       ├── hmnist_8_8_RGB.csv            (optional, not used)
       ├── hmnist_28_28_L.csv            (optional, not used)
       └── hmnist_28_28_RGB.csv          (optional, not used)

3. Run the organization script:
   Open MATLAB, navigate to the project src folder, and run:
   
   >> step1_organize_data
   
   This will create:
   - data/all_images/ (merged images from both parts)
   - data/organized_by_class/ (images sorted by diagnosis)

After Running Step 1:
---------------------
Your data/ folder will contain:

data/
├── HAM10000_images_part_1/           [Original - keep for backup]
├── HAM10000_images_part_2/           [Original - keep for backup]
├── HAM10000_metadata.csv             [Original - keep]
├── all_images/                       [Created by script - 10,015 images]
├── organized_by_class/               [Created by script]
│   ├── akiec/    (327 images)
│   ├── bcc/      (514 images)
│   ├── bkl/      (1099 images)
│   ├── df/       (115 images)
│   ├── mel/      (1113 images)
│   ├── nv/       (6705 images)
│   └── vasc/     (142 images)
└── hmnist CSV files                  [Optional - can delete if needed]

Dataset Information:
--------------------
Total Images: 10,015 dermoscopic images
Image Format: JPG (RGB color)
Image Resolution: Variable (typically 600×450 to 1024×768 pixels)
                  Will be resized to 224×224 for neural network input

Class Distribution:
- akiec (Actinic keratoses and intraepithelial carcinoma): 327 images (3.3%)
- bcc (Basal cell carcinoma): 514 images (5.1%)
- bkl (Benign keratosis-like lesions): 1099 images (11.0%)
- df (Dermatofibroma): 115 images (1.1%) - smallest class
- mel (Melanoma): 1113 images (11.1%)
- nv (Melanocytic nevi): 6705 images (67.0%) - largest class (highly imbalanced)
- vasc (Vascular lesions): 142 images (1.4%)

Metadata Information:
---------------------
The HAM10000_metadata.csv contains 10,015 rows with these columns:

- lesion_id: Unique lesion identifier
- image_id: Unique image identifier (matches .jpg filename)
- dx: Diagnosis (class label) - one of the 7 categories above
- dx_type: Method of diagnosis confirmation:
    * histo: Histopathology (biopsy)
    * follow_up: Follow-up examination
    * consensus: Expert consensus
    * confocal: Confocal microscopy
- age: Patient age (years)
- sex: Patient sex (male/female)
- localization: Anatomical location of lesion:
    * abdomen, back, chest, ear, face, foot, genital, hand, 
      lower extremity, neck, scalp, trunk, upper extremity, etc.

Dataset Characteristics:
------------------------
- Over 50% of lesions confirmed through histopathology
- Remaining cases verified through follow-up, expert consensus, or confocal microscopy
- Images collected from different populations worldwide
- Various camera equipment and imaging conditions
- Highly imbalanced dataset (nv class dominates with 67% of images)

Storage Requirements:
---------------------
- Original dataset (downloaded): ~1.9 GB
- After organization (all_images + organized_by_class): ~4 GB total
- Ensure you have at least 5 GB free space before starting

Download Time Estimates:
------------------------
- Fast internet (50+ Mbps): 5-10 minutes
- Medium internet (10-50 Mbps): 15-30 minutes
- Slow internet (<10 Mbps): 30-60 minutes

Expected Processing Time:
--------------------------
Step 1 (organize_data.m) will take approximately:
- Merging images: 5-8 minutes
- Organizing by class: 5-8 minutes
- Total: ~10-15 minutes

Verification:
-------------
After setup, verify your installation:

1. Check that organized_by_class/ contains 7 subdirectories
2. Each subdirectory should contain .jpg files
3. Total should be 10,015 images across all classes

Run this in MATLAB to verify:
>> cd('data/organized_by_class')
>> classes = {'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'};
>> for i = 1:length(classes)
       files = dir(fullfile(classes{i}, '*.jpg'));
       fprintf('%s: %d images\n', classes{i}, length(files));
   end

Expected output:
akiec: 327 images
bcc: 514 images
bkl: 1099 images
df: 115 images
mel: 1113 images
nv: 6705 images
vasc: 142 images

Dataset Citation:
-----------------
If you use this dataset in your work, please cite:

Tschandl, P., Rosendahl, C., & Kittler, H. (2018). 
The HAM10000 dataset, a large collection of multi-source 
dermatoscopic images of common pigmented skin lesions. 
Scientific Data, 5, 180161. 
https://doi.org/10.1038/sdata.2018.161

Dataset License:
----------------
The HAM10000 dataset is licensed under CC BY-NC 4.0
(Creative Commons Attribution-NonCommercial 4.0 International)

This means:
✓ You can use it for academic/research purposes
✓ You can modify and build upon the dataset
✗ You cannot use it for commercial purposes without permission
✓ You must give appropriate credit to the original authors

Troubleshooting:
----------------

Problem: "Cannot download from Kaggle"
Solution: You must be logged in to Kaggle with a verified account.
          Kaggle requires phone verification for downloads.

Problem: "Download is very slow"
Solution: Try downloading during off-peak hours or use Kaggle API:
          pip install kaggle
          kaggle datasets download -d kmader/skin-cancer-mnist-ham10000

Problem: "Images not found" error in MATLAB
Solution: Ensure folders are in data/ directory and named exactly:
          - HAM10000_images_part_1
          - HAM10000_images_part_2
          (folder names are case-sensitive)

Problem: "Metadata CSV not found"
Solution: Ensure HAM10000_metadata.csv is directly in data/ folder

Problem: "Out of disk space during organization"
Solution: The organization process temporarily requires ~4GB.
          Free up space and run step1_organize_data.m again.

Problem: "Images appear corrupted"
Solution: Re-download the dataset. Extraction errors can corrupt files.
          Use 7-Zip or WinRAR for reliable extraction.