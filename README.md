# Willow: A Medicinal Tree
Helping patients access the prescription medication they need.


## Table of Contents
- [Problem Description](#problem)
- [Data Description](#data)
- [Model](#model)
- [Prototype](#tool)
- [Results](#result)
- [Using this Repository](#use)
- [Authors](#author)


## Problem Description <a name="problem"></a>
We are hoping to solve the following problems:
1. Can a claim’s rejection be predicted using claim data? If so, what factors tend to influence the rejection of a claim?
2. If a claim is rejected, can the reason for rejection (rejection code) be predicted?
3. Can the approval of a PA be predicted using claim data? If so, what factors tend to influence the approval of a PA?


## Data Description <a name="data"></a>
All of the data were provided by [CoverMyMeds](https://www.covermymeds.com/main/). The data consists of four tables of simulated data pertaining to patients’ access to drugs and prior authorizations (PA) due to rejection by the payer. There are ~1.3 million claims across three years from January 1, 2017 to December 31, 2019. 

The simulated pharmacy claim-level data provided is for three drugs (A, B, C) and four payers (417380, 417614, 417740, 999001). There are ~556k rejected claims that required a PA. There are three rejection codes provided; “70” for a drug that is not covered by the plan and not on formulary, “75" for a drug on the formulary that does not have preferred status and requires a PA, and “76” for a drug that is covered but the plan limitations have been exceeded. 

The simulated PA data contains four binary categories indicating whether the patient has the correct diagnosis (80% of PAs), has tried and failed a generic alternative (50% of PAs), if the patient has an associated contraindication (20% of PAs), and whether the PA was approved (73% of PAs).


## Stakeholders
The main stakeholders the app is intended for are developers and analysts working on the [electronic prior authorization (ePA) product](https://www.covermymeds.com/main/solutions/payer/epa/).


## Modeling Approach <a name="model"></a>
We created classification models for our three problems using logistic regression, decision tree, and random forest.
The models were evaluated by comparing recall, precision, accuracy, and ROC AUC using 5-fold cross validation on the training data. Precision was prioritized because false positives (approving a claim that will actually be rejected) was deemed to be worse than a false negative, because predicing an approval when it will later be denied can cause prescription abandonment.
- Decision tree and random forest performed equally well and better than logistic regression.
- We have determined that decision trees are the best combination of perfomance, simplicity, and interpretability.
- We are using the `max_leaves` hyperparameter to tune the decision trees and prevent overfitting.
- A 20% test split was used to evaluate model performance.

## We created an app! <a name="tool"></a>
http://willow-erdos.herokuapp.com/ ([backup deployed using Streamlit's sharing service](https://share.streamlit.io/nickmacro/erdos-covermymeds-project/main/app.py)) The app serves as an explanation of the model, data used by the model, and a prototype of the model in action.

## Results <a name="result"></a>
### Claim Approval
Each payer has a unique set of drugs that it approves without prior authorization. Payer 999001 accepts all claims, regardless of drug at a rate of 90%. The other payers only accept a single drug at a rate of 90% and reject the remaining two drugs. The following table details the drug-payer relationship for claim approval rates. 

|Payer / Drug | A | B | C |
| - | - | - | - |
|999001|90%|90%|90%|
|417380|0%|90%|0%|
|417614|0%|0%|90%|
|417740|90%|0%|0%|

Details in [claims-exploration notebook](https://github.com/NickMacro/erdos-covermymeds-project/blob/5126ecdd9928a10eb70394cf14a67c160cd3f612/exploration/2021-05-14_nm_claims-exploration.ipynb).

### Reason for Rejection
When a claim is denied, a rejection code is provided that indicates the reason for denial. The claims that are approved in the **Claim Approval** table, are rejected with a code **76**. The drugs that are not covered by the payer are rejected with either code **70** or code **75**. The table below displays the formulary for each payer.

|Payer / Drug | A | B | C |
| - | - | - | - |
|999001|76|76|76|
|417380|75|76|70|
|417614|70|75|76|
|417740|76|70|75|

- **70** not covered by the plan and not on formulary
- **75** on the formulary that does not have preferred status and requires a PA
- **76** covered but the plan limitations have been exceeded. 

Details in [pa-exploration notebook](https://github.com/NickMacro/erdos-covermymeds-project/blob/5126ecdd9928a10eb70394cf14a67c160cd3f612/exploration/2021-05-12_nm_pa-exploration.ipynb).

### Prior Authorization Approval
Prior authorizations with rejection codes of 70 have the lowest approval rate (33% - 58%), rejection code 75 with the highest approval rate (83% - 99%), and rejection code 76 in the middle (64% - 96%).

A correct diagnosis for the associated drug proved an overall 4% increase in PA approval rate. Trying an failing a generic alternative provides an overall 11% increase in PA approval rate. A contraindication decreases the overall approval rate by 25%. These changes in PA approval rate depend on drug-payer combination.

Details in [pa-exploration notebook](https://github.com/NickMacro/erdos-covermymeds-project/blob/5126ecdd9928a10eb70394cf14a67c160cd3f612/exploration/2021-05-12_nm_pa-exploration.ipynb).

### Time Independence of Approval Rates
The approval rate of both claims and prior authorizations do not change with time. Details in [claims-time-exploration notebook](https://github.com/NickMacro/erdos-covermymeds-project/blob/5126ecdd9928a10eb70394cf14a67c160cd3f612/exploration/2021-05-14_nm_claims-time-exploration.ipynb) and [pa-time-exploration notebook](https://github.com/NickMacro/erdos-covermymeds-project/blob/5126ecdd9928a10eb70394cf14a67c160cd3f612/exploration/2021-05-14_nm_pa-time-exploration.ipynb).

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

The user environment can be installed using `conda`:

`conda env create -f user-environment.yml`

Streamlit can be installed using `pip`:

`pip install streamlit`

The hosted Streamlit applications are automatically updated when the GitHub repository is changed.

## Authors <a name="author"></a>
This project was created by [Nick Macro](https://www.linkedin.com/in/nickmacro/), [Sandrine Müller](https://www.linkedin.com/in/sandrinermuller/), and [Tomas Kasza](https://www.linkedin.com/in/tomas-kasza/).
