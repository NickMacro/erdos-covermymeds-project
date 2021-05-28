# erdos-covermymeds-project
Repository for a project in the Erdős Institute spring 2021 bootcamp CoverMyMeds corporate member project.

## Data Description
The data consists of four tables of data pertaining to patients’ access to drugs and prior authorizations (PA) due to rejection by the payer. There are ~1.3 million claims across three years from January 1, 2017 to December 31, 2019. The pharmacy claim-level data provided is for three drugs (A, B, C) and four payers (417380, 417614, 417740, 999001). There are ~556k rejected claims that require a PA. There are three rejection codes provided; “70” for a drug that is not covered by the plan and not on formulary, “75" for a drug on the formulary that does not have preferred status and requires a PA, and “76” for a drug that is covered but the plan limitations have been exceeded. The PA data contains four binary categories indicating whether the patient has the correct diagnosis (80% of PAs), has tried and failed a generic alternative (50% of PAs), if the patient has an associated contraindication (20% of PAs), and whether the PA was approved (73% of PAs).

## Problem Description
We are hoping to solve the following problems:
1. Can the approval of a PA be predicted using claim and PA data? If so, what factors tend to influence the approval of a PA?
2. Can a claim’s rejection be predicted using claim and PA data? If so, what factors tend to influence the rejection of a claim?

## Stakeholders
The main stakeholders the app is intended for are developers and analysts working on the electronic prior authorization (ePA) product.(https://www.covermymeds.com/main/solutions/payer/epa/)

## Modeling Approach
We created classification models for our three problems using logistic regression, decision tree, and random forest.
The models were evaluated by comparing recall, precision, and accuracy using 5-fold cross validation on the training data. Due to hyperparameter tuning being incomplete for some models, we have not evaluated the models on the test data.
- Decision tree and random forest performed equally well and better than logistic regression.
- We have decided to use decision tree instead of logistic regression and random forest because a decision tree is easily interpreted and less complex than the logistic regression and random forest models.
- We are using min_impurity_decrease and ccp_alpha to tune the decision trees and prevent overfitting.
- We split the data across payers to evaluate performance for each payer and evaluate model performance using ROC AUC.

## We created an app! 
http://willow-erdos.herokuapp.com/
The app serves as an explanation of the model, data used by the model, and a prototype of the model in action.
