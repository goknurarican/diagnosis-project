# Automatic Medical Diagnosis (DDXPlus)

## Setup

```bash
git clone https://github.com/goknurarican/diagnosis-project.git
cd diagnosis-project
python -m venv venv
source venv/bin/activate      #or windows: venv\Scripts\activate
pip install -r requirements.txt


How LightGBM Model Works on DDXPlus
We start with the DDXPlus synthetic patient data (around 1.3M records of age, sex, ground-truth disease, and lists of symptoms). Our goal is to train a model that predicts a patient’s disease from their symptoms without peeking at any “cheat” columns.

a) Data Cleanup
-Removed columns that would leak the answer (initial evidence and differential diagnosis).
-Filled missing age with the median, map sex to 0/1.
-Converted each symptom list into a one-hot (multi-hot) vector.

b)Feature masking and noise
-Identified the top 100 most frequent symptom codes and mask them out—this prevents the model from learning to simply key on the most common signals.
-Randomly dropped 40 % of the remaining symptom codes on each patient to force the model to learn robust patterns.

c)Features
-Demographics:age (binned into 4 ranges) + sex (0/1).
-Symptoms:sparse multi-hot vectors over the ~400 remaining symptom codes.
-We saved this as a scipy CSR matrix (.npz) plus a NumPy array of disease labels (.npy).

d)Training with LightGBM
-We run 5-fold stratified cross-validation, so each fold sees 80 % of train data and validates on 20 %.
-LightGBM hyperparameters are tuned for regularization (e.g. feature_fraction, bagging_fraction, shallow trees) to avoid overfitting.
-Early stopping on validation loss after 20 rounds prevents wasting time on models that no longer improve.

e)Evaluation
-On the held-out test set, we achieved ~88.6 % accuracy and ~0.89 macro-F1.
-A confusion matrix highlights which diseases the model still confuses—those become prime candidates for further feature engineering or clinician review.

