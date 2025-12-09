# Finguard AI
## Task1
* The dataset has been taken from kaggle, using kagglehub. It contains around 5,00,000 entries of data. the data contains 11 columns. The dataset is distinguishing all the transsacrtions into fraud transactions and safe transactions. The dataset is imbalanced. 
* Dataset takes into account the, transaction time and upi numbers.
* I have compared the fraud_transaction count vs safe transaction counts. Also does correlation matrix between features. 
* I have compared between hour of transaction and fraud_risk. & boxplot to find the distribution of quartets of amount for fraud_risk=1 and fraud_risk=0.
* It has seaborn heatmap plots also.  
## Task2

* Due to troubles faced with cpu on local system, I moved to GPU on colab.
* Thus AI has been used to implement FILE2, FILE3 (first 70% or so of both the files. i.e., the CPU implementation of these has been made in File 1 by myself.)
* File2 uses manual tuning of hyperparameters, using values from plots.
* File3 uses Randomized SearchCV to find the hparams. Also (the evaluation and SHAP analysis is mostly self implemented.)
* Sorry for Creating the Confusion, if any.

### Task 2 Objective:
    1. The objective of task2 is to find a best suited model for our given task.
    2. The objective is also to find the credentials of transaction situations which affected the fraudulance found in the data. 

### Short description of each model
    1. **Random Forest** : implemented from sklearn(cpu) and cuml(gpu T4) is an trees ensemble where multiple decision trees run in parallel and majority vote is considered as output.
    2. **XG Boost** implemented using xgboost library. works based on sequential tree building working on previous trees' mistakes.
    3. **SVM** :Linear SVMs work by finding an optimal hyperplane in high-dimension space. Implemented using cuml library for gpu and sklearn library for cpu.

### Hyperparameter justification:
    1. The hyperparameters have been chosen keeping in mind the tradeoff between accuracy and computation.
    2. Generally tried to have higher accuracy without overfitting.

### Explaination of choice to work with imbalanced-data:
    1. The training of models and working with models while imbalanced data is there is absurd.
    2. Thus the data has been normalized first and then it has been fit into SMOTE to work with imbalanced training data.
    
### Metric Interpretations:
    1. The Interpretations are not just based on the accuracy, but also precision and content of confusion matrix.
    2. AUC-ROC gives a number between 0 and 1, giving the idea of robustness of the model. If the value is higher, that means the model us working better.
    3. thus, pick a model with higher auc-roc.
#### The Confusion Matrix:
```bash
Evaluating Random Forest (GPU)...

--- Random Forest (GPU) Classification Report ---
              precision    recall  f1-score   support

           0       0.84      0.97      0.90     42050
           1       0.18      0.04      0.07      7950

    accuracy                           0.82     50000
   macro avg       0.51      0.50      0.48     50000
weighted avg       0.74      0.82      0.77     50000

Confusion Matrix:
[[40617  1433]
 [ 7630   320]]

Evaluating XGBoost (GPU)...

--- XGBoost (GPU) Classification Report ---
              precision    recall  f1-score   support

           0       0.84      1.00      0.91     42050
           1       0.25      0.00      0.00      7950

    accuracy                           0.84     50000
   macro avg       0.55      0.50      0.46     50000
weighted avg       0.75      0.84      0.77     50000

Confusion Matrix:
[[42047     3]
 [ 7949     1]]

Evaluating LinearSVC (GPU)...

--- LinearSVC (GPU) Classification Report ---
              precision    recall  f1-score   support

           0       0.84      0.99      0.91     42050
           1       0.23      0.02      0.03      7950

    accuracy                           0.84     50000
   macro avg       0.54      0.50      0.47     50000
weighted avg       0.75      0.84      0.77     50000

```
### SHAP_Results:
    1. SHAP determines a dominance of each transaction details on fraudulence of the transactions.
    2. This is visualised using the beeswarm graph, waterfall graph and bar graph.
```bash
Winner: XGBoost (GPU) (AUC=0.5755)

Starting SHAP explanation for XGBoost (GPU)...

Mean Absolute SHAP Values (Decreasing Order of Importance):

     Feature  Mean Absolute SHAP Value
 trans_month                  0.328607
    category                  0.304747
  trans_hour                  0.266090
         zip                  0.168878
   trans_day                  0.124649
         age                  0.089631
       state                  0.063210
trans_amount                  0.046219
  upi_number                  0.031685
  trans_year                  0.000000

```
### Final result:
    1. In terms of AUC-ROC, XGBoost wins.
    2. transaction Month, category and transaction hour are shown to be the dominating factor whether a transaction in risk or not.

## References & Resources

* **[ANDREW NG | Coursera](https://github.com/greyhatguy007/Machine-Learning-Specialization-Coursera/tree/main/)**
* **[SHAP DOCUMENTATION](https://youtu.be/L8_sVRhBDLU?si=PnCji7Rgn7wtDQBY)** 
* **[EVALUATION METRICS](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)**
