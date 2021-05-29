# Willow: A Medicinal Tree
Helping patients access the prescription medication they need.


## Table of Contents
- [Problem Description](#problem)
- [Data Description](#data)
- [Model](#model)
- [Prototype](#tool)
- [Using this Repository](#use)


## Problem Description <a name="problem"></a>
We are hoping to solve the following problems:
1. Can a claim’s rejection be predicted using claim data? If so, what factors tend to influence the rejection of a claim?
2. If a claim is rejected, can the reason for rejection (rejection code) be predicted?
3. Can the approval of a PA be predicted using claim data? If so, what factors tend to influence the approval of a PA?


## Data Description <a name="data"></a>
The data consists of four tables of data pertaining to patients’ access to drugs and prior authorizations (PA) due to rejection by the payer. There are ~1.3 million claims across three years from January 1, 2017 to December 31, 2019. 

The pharmacy claim-level data provided is for three drugs (A, B, C) and four payers (417380, 417614, 417740, 999001). There are ~556k rejected claims that required a PA. There are three rejection codes provided; “70” for a drug that is not covered by the plan and not on formulary, “75" for a drug on the formulary that does not have preferred status and requires a PA, and “76” for a drug that is covered but the plan limitations have been exceeded. 

The PA data contains four binary categories indicating whether the patient has the correct diagnosis (80% of PAs), has tried and failed a generic alternative (50% of PAs), if the patient has an associated contraindication (20% of PAs), and whether the PA was approved (73% of PAs).


## Stakeholders
The main stakeholders the app is intended for are developers and analysts working on the [electronic prior authorization (ePA) product](https://www.covermymeds.com/main/solutions/payer/epa/).


## Modeling Approach <a name="model"></a>
We created classification models for our three problems using logistic regression, decision tree, and random forest.
The models were evaluated by comparing recall, precision, and accuracy using 5-fold cross validation on the training data. Precision was prioritized because false positives (approving a claim that will actually be rejected) was deemed to be worse than a false negative, because predicing an approval when it will later be denied can cause prescription abandonment.
- Decision tree and random forest performed equally well and better than logistic regression.
- We have determined that decision trees are the best performing, simplest, and most interpretable.
- We are using the `max_leaves` hyperparameter to tune the decision trees and prevent overfitting.
- A 20% test split was used to evaluate model performance.

## We created an app! <a name="tool"></a>
http://willow-erdos.herokuapp.com/ [backup](https://share.streamlit.io/nickmacro/erdos-covermymeds-project/main/app.py)
The app serves as an explanation of the model, data used by the model, and a prototype of the model in action.

## Using this Repository <a name="use"></a>
This repository contains the data and code used to create the prototype, models, and figures used in this project. The contents are organized as follows:
```
📁 data : all of the data used for the project
   📁 raw : raw data for the project (this should never be changed)
   📁 processed : training and test data
📁 documents : documents describing the project
📁 exploration : notebooks used for exploratory data analysis
📁 models : notebooks, joblib files, and figures related to models
    📁 saved-models : persistent joblib files for the final models
    📁 saved-model-figures : figures for the final models
📁 tasks : notebooks used for miscellaneous tasks

📄 Procfile : used by Heroku deployment of Streamlit app
📄 app.py : python script used to create the Streamlit app
📄 requirements.txt : dependencies used by Heroku for the Streamlit app deployment
📄 user-environment.yml : dependencies used in the notebooks
```
