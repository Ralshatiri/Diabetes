# Diabetes Classification 

This repository contains 
1) data discovery + preprocessing  
2) decision tree model development + evaluation

The project uses a diabetes dataset and builds Decision Tree classifiers using both **Gini** and **Entropy** criteria, comparing performance across multiple train/test splits.

---

## Dataset

- **Source:** Kaggle — Diabetes Dataset (2019)  
- **Rows:** 952  
- **Columns:** 18  
- **Target label:** `Diabetic` (yes / no)

Main features include: Age, Gender, BMI, Sleep, SoundSleep, HighBP, Family_Diabetes, Smoking, Alcohol, Stress, BPLevel, etc.

---

## Repository Files

### `01_Data_Discovery_and_Preprocessing.ipynb`
What this notebook does:
- Loads the raw dataset (`diabetes_dataset__2019.csv`)
- Displays dataset info (shape, columns, types, summary stats)
- Performs EDA with visualizations (gender, age, diabetic distribution, etc.)
- Handles missing values
- Encodes categorical features (Label Encoding) to compute correlation/heatmap
- Detects outliers using boxplot + density plots
- Normalizes selected numeric columns using **MinMaxScaler**
- Exports the final cleaned dataset as:
  `Processed_dataset.csv`

---

### `02_Model_Development_and_Evaluation.ipynb`
What this notebook does:
- Loads `Processed_dataset.csv`
- Encodes categorical columns using **LabelEncoder**
- Splits data into 3 partitions for comparison:
  - 70% train / 30% test
  - 60% train / 40% test
  - 80% train / 20% test
- Trains Decision Tree classifiers using:
  - **Entropy**
  - **Gini**
- Evaluates performance using:
  - Accuracy
  - Confusion Matrix
  - Sensitivity (Recall), Specificity, Precision
- Visualizes the decision trees using `sklearn.tree.plot_tree`
- Demonstrates prediction on a **new sample input** after encoding + scaling.

