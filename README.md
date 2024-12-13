# CSCSI475 Group 6: Predicting Study Reproducibility Using Common Experimental Features

## Description
Our project attempts to leverage the FReD dataset in order to predict whether statistical studies are replicable. This repository contains our work in this problem, with the data cleanup, EDA, and model codes all included. 

## How to reproduce results:


### 1. Random Forest
Since we used grid search to optimize our hyperparameters, the simplest way to reproduce our results is by creating a random forest algorithm that uses the optimal parameters we found. These can be found in the notebook's output, but for transparency's sake we'll include them here as well:

Best Parameters: `{'bootstrap': False, 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}`

So it can be set up the following way:
```
from sklearn.ensemble import RandomForestClassifier

best_params = {'bootstrap': False, 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

model = RandomForestClassifier(**best_params)
```

All the code for the Random Forest model can be found in `model_rf.ipynb`.

### 2. Gradient Boosting Machine
Once again, since we used Optuna here, it is much easier to load our model's best parameters. These can be found in `models/lgbm_0.76_notags.txt`. It can be loaded the following way:

```
import lightgbm as lgb

lgbm_model = lgb.Booster(model_file='models/lgbm_0.76_notags.txt')
```

All code for the Gradient Boosting Machine's training and evaluation is included in `model_lightgbm.ipynb`, and the plotting code is also included in `plotting_lightgbm.ipynb`.


### 3. Logistic Regression

The logistic regression model was optimized using grid search. The optimal parameters are: `{'C': 100, 'penalty': 'l1', 'solver': 'liblinear'}`.

So it can be set up the following way:
```
from sklearn.linear_model import LogisticRegression
best_params = {'C': 100, 'penalty': 'l1', 'solver': 'liblinear'}
lr = LogisticRegression(**best_params)
```

All the code for the logistic regression model can be found in `model_logreg.ipynb`.



## Dependencies
All dependencies used can be found in `requirements.txt`.

## Credits
* Regina: 
  * Data cleaning
  * LightGBM model & plotting
  * Data selection, and tree-based models methodology in report
* Brian: 
  * Data search
  * Report intro, background, and conclusion.
* Matija: 
  * Random forest model
  * Results section in report
  * Team coordination.
* Darren: 
  * EDA
  * Logistic regression model
  * Logistic regression methodology in report.

