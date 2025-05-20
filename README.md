# Automatic Medical Diagnosis (DDXPlus)


This project focuses on building a machine learning system for automatic disease prediction using structured patient data. The aim is to assist in the development of intelligent clinical decision support systems by predicting the most likely disease based on patient demographics and reported symptoms.

## Dataset: DDXPlus

The project uses the **DDXPlus** dataset, a large-scale synthetic medical dataset containing approximately **1.3 million patient records**. Each patient record includes:

- **Demographics**: Age and sex  
- **Pathology**: The correct disease (target variable)  
- **Symptoms (Evidences)**:  
  - Binary (e.g., fever: yes/no)  
  - Categorical (e.g., pain intensity: mild/moderate/severe)  
  - Multi-choice (e.g., pain locations: multiple selections)  
- **Differential Diagnosis**: Excluded from training to prevent data leakage

Dataset is split into:
- `train.csv` (~1.03M records)  
- `validate.csv` (~132K records)  
- `test.csv` (~135K records)  
Additional files:  
- `release_evidences.json`: Defines symptom types and options  
- `release_conditions.json`: Disease definitions and ICD-10 mappings

---

## Models Used

Two models were developed and evaluated:

### LightGBM  
A gradient boosting model based on decision trees. Preferred for its speed, efficiency, and natural compatibility with sparse, categorical, and imbalanced data.

- 5-fold stratified cross-validation
- Handles class imbalance using `class_weight='balanced'`
- Saved as `.txt` models per fold

###  MLP (Multilayer Perceptron)  
A neural network model used for comparison with deep learning-based methods. Sparse inputs are converted to dense before training.

- 5-fold stratified cross-validation  
- Training loss curves saved as `.png`  
- Models saved as `.pkl` per fold

---


### Script Explanations

This section provides a brief description of each Python script used in the project:

#### `process_data.py`
- Cleans and preprocesses the raw CSV files  
- Handles missing values (e.g., age, sex)  
- Drops leakage columns (`DIFFERENTIAL_DIAGNOSIS`, `INITIAL_EVIDENCE`)  
- Normalizes and encodes features  
- Saves the cleaned data as `.parquet` files

#### `feature_build.py`
- Converts cleaned `.parquet` files into model-ready formats  
- Bins age, encodes sex, and applies multi-hot encoding to symptoms  
- Masks top evidence codes and injects random noise for robustness  
- Outputs: `X.npz` (features) and `y.npy` (labels)

#### `train.py`
- Trains **LightGBM** models using 5-fold stratified cross-validation  
- Automatically handles class imbalance  
- Saves a separate model for each fold (e.g., `fold0.txt`)  
- Prints per-fold metrics: Accuracy, F1-Score, MCC

#### `train_mlp.py`
- Trains **Multilayer Perceptron (MLP)** models using 5-fold CV  
- Converts sparse features to dense format  
- Saves each fold model as `.pkl`  
- Plots training loss curves for each fold

#### `eval.py`
- Evaluates a single **LightGBM** model on the test set  
- Reports Accuracy, F1-Score, and MCC  
- Optionally saves a confusion matrix plot as an image

#### `eval_mlp.py`
- Evaluates all **5 MLP fold models** on the test set  
- Reports average metrics across all folds  
- Optionally saves individual confusion matrices for each fold

#### `explore_ddxplus.py`
- Performs **Exploratory Data Analysis (EDA)** on the dataset  
- Generates plots for:
  - Age distribution  
  - Sex distribution  
  - Top 20 pathologies  
  - Top 20 evidence codes  
- Outputs a text summary with dataset statistics and missing value analysis


## Evaluation

Each model is evaluated using:

- **Accuracy**
- **Macro F1-Score**
- **Matthews Correlation Coefficient (MCC)**
- **Confusion Matrix (optional visualizations)**

All metrics are averaged across folds to assess stability and generalization.

---


### How to run the project
```bash
git clone https://github.com/goknurarican/diagnosis-project.git
cd diagnosis-project
```

```bash
python -m venv venv
source venv/bin/activate      #or windows: venv\Scripts\activate
pip install -r requirements.txt
```

```bash
#Preprocess raw dataset
python process_data.py --input-dir ./ddxplus --output-dir ./cleaned_ddxplus

````

```bash
# Train set
python feature_build.py --input ./cleaned_ddxplus/train_clean.parquet --output-dir ./features

# Validation set
python feature_build.py --input ./cleaned_ddxplus/validate_clean.parquet \
  --output-dir ./features \
  --ev2id ./features/ev2id.json \
  --cond2id ./features/cond2id.json

# Test set
python feature_build.py --input ./cleaned_ddxplus/test_clean.parquet \
  --output-dir ./features \
  --ev2id ./features/ev2id.json \
  --cond2id ./features/cond2id.json

````

```bash
#Train LighGBM
python train.py --x ./features/train_X.npz --y ./features/train_y.npy --output ./models/lgb


````


```bash
#Train MLP
python train_mlp.py --x ./features/train_X.npz --y ./features/train_y.npy --output ./models/mlp

````


```bash
#Evaluate LightGBM (single fold)
python eval.py \
  --model ./models/lgb/fold0.txt \
  --test_X ./features/test_X.npz \
  --test_y ./features/test_y.npy \
  --plot_cm

````

```bash
#Evaluate All MLP Folds
python eval_mlp.py \
  --model-dir ./models/mlp \
  --test_X ./features/test_X.npz \
  --test_y ./features/test_y.npy \
  --plot_cm \
  --cm_out_dir ./models/mlp/confusion_matrices

````


