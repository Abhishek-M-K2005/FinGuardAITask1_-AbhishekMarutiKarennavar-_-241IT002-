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

References:
    * [ANDREW NG | Coursera](https://github.com/greyhatguy007/Machine-Learning-Specialization-Coursera/tree/main/ )
    * [SHAP DOCUMENTATION](https://youtu.be/L8_sVRhBDLU?si=PnCji7Rgn7wtDQBY)
    * [EVALUATION METRICS](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)