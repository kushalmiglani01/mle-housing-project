# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest
 - Grid Search CV
 - Randomized Search CV
The model implemented for the final prediction:

 - SVM
 - RFE for feature selection using Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To setup the mle-dev environment
 - Run the following command
```
conda env create -f env.yml
```

## To activate the mle-dev environment
 - Run the following command

```
conda activate mle-dev
```

## To excute the scrip
 - There are three main scripts in the housingmodel package
    - ingest_data.py
    - train.py
    - score.py
    
 - For help
```
python src/housingmodel/<scriptname.py> -h
```
