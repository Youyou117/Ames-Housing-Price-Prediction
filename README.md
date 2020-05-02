# Airbnb-House-Prediction
#### Background:

Ask a home buyer to describe their dream house, and they will probably come up with many criterias, based on which, we want to help the home buyers to predict their dream house price.

We obtained our dataset from Kaggle. com, which includes 1 response variable (sales price) and 79 explanatory variables describing residential homes in Ames, Iowa. 

#### Objective Goal: 

●	Our goal is to predict house prices with labels and attributes of Ames Housing dataset.

●	Our models are evaluated on the Root-Mean-Squared-Error (RMSE) between the log of the SalePrice predicted by our model, and the log of the actual SalePrice. 

#### Model Key Features: 

●	Cross Validation: Using 12-fold cross-validation

●	Models: On each run of cross-validation we fit 3 models (ridge, random forest and xgboost)
