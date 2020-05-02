# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor

# Misc
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


# load data
# Jinxin
train = pd.read_csv(r"C:\Users\18612\Desktop\UMD_MSBA\758b\Projec\house-prices-advanced-regression-techniques\train.csv")
test = pd.read_csv(r"C:\Users\18612\Desktop\UMD_MSBA\758b\Projec\house-prices-advanced-regression-techniques\test.csv")

# Yutian Luo
train  = pd.read_csv(r"C:\B proj\train.csv")
test = pd.read_csv(r"C:\B proj\test.csv")

#Hayley
train = pd.read_csv(r"C:\Users\10459\OneDrive\Desktop\HL\Maryland\Class\BUDT 758B MW11am\Final Project\train.csv")
test = pd.read_csv(r"C:\Users\10459\OneDrive\Desktop\HL\Maryland\Class\BUDT 758B MW11am\Final Project\test.csv")

#Xiaoyou Zhou
train = pd.read_csv(r"C:\Users\ZXY0117\Desktop\Course_S2\Big Data and AI for Business(B)\Project\Data\train.csv")
test = pd.read_csv(r"C:\Users\ZXY0117\Desktop\Course_S2\Big Data and AI for Business(B)\Project\Data\test.csv")

# Allen
train = pd.read_csv(r"C:\Users\super\Desktop\courses\s2\BUDT758B\project\train.csv")
test = pd.read_csv(r"C:\Users\super\Desktop\courses\s2\BUDT758B\project\test.csv")

#Siyuan
train = pd.read_csv(r"C:\Users\Admin\Desktop\lsy\data\bigdata_project\train.csv")
test = pd.read_csv(r"C:\Users\Admin\Desktop\lsy\data\bigdata_project\test.csv")

## EDA

    # Preview the data
train.head()

#Figure 1
#Plot SalePrice
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(15, 12))
sns.set_color_codes(palette='pastel')
sns.distplot(train['SalePrice'], color="g")

ax.set(ylabel="Price Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
plt.show()

#Comments:
# from the plot we can see that sale price obeys right skewed distribution, which aligns with our common sense


#Figure 2
#plot the correlations between features and label.
featuresCorrelation = train.corr() 
plt.subplots(figsize=(40, 25))
ax = sns.heatmap(featuresCorrelation, cmap="YlGnBu", square=True)
ax.set_xticklabels([x.get_text() for x in ax.get_xticklabels()], rotation=90)
plt.show()

#Comments:
#From the heatmap we can see that "GarageYrBlt" and "YearBuilt" are highly correlated, \
#"GarageCars" and "GarageArea" , "TotalBsmtSF" and "1stFlrSF" are also highly correlated. \
#Therefore, when building our model, we will delete one feature of each pair. 

#Figure 3 sale price ~ overall quality (Rates the overall material and finish of the house)
#plot the correlation between SalePrice and Overall Quallity
sale_overallQ = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)
fig, ax = plt.subplots(figsize=(10, 8))
fig = sns.boxplot(x=train['OverallQual'], y="SalePrice", data=sale_overallQ, color= 'g')
plt.show()

#Comments:
# Based on the overall quality to sale price, we have several findings:
# 1. when the overall quality increased, the sale price will increase;
# 2. when the overall quality increased, the range of sale price will increase as well;
# 3. almost all outliers are upward anomalies (above the upper bound), 
# indicating that all alternatives that are not following the mainstream are high prices,
# not the floor price. The sale side is the dominant side in real estate sale industry.

#Figure 4:    SalePrice ~ Year Built
sale_YearBuilt = pd.concat([train['SalePrice'], train['YearBuilt']], axis=1)
fig, ax = plt.subplots(figsize=(12, 8))
fig = sns.boxplot(x='YearBuilt', y="SalePrice", data=sale_YearBuilt, color='g')
xtick_label = fig.get_xticklabels()
xtick_label = [x.get_text() if int(x.get_text())%5 == 0 else '' for x in xtick_label]
fig.set_xticklabels(xtick_label, rotation=30)
plt.show()

#Comments:
# Based on the overall quality to sale price, we have several findings:
# 1. generally, the sale price is increasing stablely over years. However, some ancient houses might have higher value.
# 2. 3 points of starting increase: 
# a. 1950 was the midpoint of the first industry boom after the end of World War II, when single-family home construction reached a record high. 
# b. 1972 was the midpoint of the second industry boom after the end of the Second World War. 
#    During that period, multi-family houses showed rapid growth, and the number of newly started houses reached a historical peak. 
# c. 1993 was the third industry boom period, and single-family residential sales grew rapidly again.
# 3. 3 points of starting decrease:
# a. In the US state of Florida from 1923 to 1926, this real estate speculation frenzy caused the Wall Street stock market to collapse, 
#    and real estate prices also changed dramatically.
# b. In 1981, the world's first economic crisis broke out in the 1980s, and the unemployment rate in the United States increased dramatically. 
#    People are full of negative attitudes towards the market and the volume of real estate transactions has shrunk dramatically.
# c. When the subprime mortgage crisis broke out in 2008, more and more people could no longer afford housing loans. 
#    They had to sell their houses at a low price or sell them at a very low price after being recovered by banks.


# Remove the Ids since they are irrelevent to prediction
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

# Feature Eng
# sales price is right skewed, transform by log(1+x)
train["SalePrice_norm"] = np.log1p(train["SalePrice"])


# Split features and labels
trainLabels = train['SalePrice'].reset_index(drop=True)
trainFeatures = train.drop(['SalePrice'], axis=1)
testFeatures = test

# concatenate train and test features in order to do the data manipulation together
all_features = pd.concat([trainFeatures, testFeatures]).reset_index(drop=True)
all_features 
# SOLVE missing values

# check the missing condition of each value
missing = all_features.apply(lambda x: round(x.isnull().mean()*100,2))
missing.sort_values(ascending=False).head(15)



# change data types of each column to proper ones
# check all the data type of each column
pd.set_option('display.max_rows', 100)
all_features.dtypes

all_features['GarageYrBlt'] = all_features['GarageYrBlt'].astype(str)
all_features['MSSubClass'] = all_features['MSSubClass'].astype(str)
all_features['YrSold'] = all_features['YrSold'].astype(str) 
all_features['MoSold'] = all_features['MoSold'].astype(str)
all_features['YearBuilt'] = all_features['YearBuilt'].astype(str)
all_features['YearRemodAdd'] = all_features['YearRemodAdd'].astype(str)


# deal with numeric values
all_features['Exterior1st'] = all_features['Exterior1st'].fillna(all_features['Exterior1st'].mode()[0])
all_features['Exterior2nd'] = all_features['Exterior2nd'].fillna(all_features['Exterior2nd'].mode()[0]) 
all_features['SaleType'] = all_features['SaleType'].fillna(all_features['SaleType'].mode()[0])
all_features['MSZoning'] = all_features['MSZoning'].fillna(all_features['MSZoning'].mode()[0])
all_features['LotFrontage'] = all_features['LotFrontage'].fillna(all_features['LotFrontage'].mean())

# deal with objects 
all_features['Functional'] = all_features['Functional'].fillna('Typ')
all_features['Electrical'] = all_features['Electrical'].fillna("SBrkr")
all_features['KitchenQual'] = all_features['KitchenQual'].fillna("TA")

object_cols = []
for i in all_features.columns:
    if all_features[i].dtype == object:
        object_cols.append(i)
all_features.update(all_features[object_cols].fillna('Not Exist'))

# NA means a house don't have a feature, so we fill with don't exist
for colname in ['BsmtQual', 'BsmtCond',
               'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
                ,'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    all_features[colname] = all_features[colname].fillna('Not Exist')
for colname in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_features[colname] = all_features[colname].fillna(0)

# fill all the na in numeric features to 0, since they don't exist
numeric_dtypes = ['float16', 'float32', 'float64', 'int16', 'int32', 'int64',]
numeric_cols = []
for i in all_features.columns:
    if all_features[i].dtype in numeric_dtypes:
        numeric_cols.append(i)
all_features.update(all_features[numeric_cols].fillna(0))

# normalize all numeric features using min-max scale
all_features[numeric_cols] = all_features[numeric_cols].apply(lambda x:(x-x.min())/(x.max()-x.min()))

# Feature Engineering
all_features['Total_Home_Quality'] = (all_features['OverallQual'] + all_features['OverallCond']) / 2
all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']
all_features['YrBltAndRemod'] = all_features['YearRemodAdd'].astype(np.uint16) - all_features['YearBuilt'].astype(np.uint16)
all_features['RemodAdd_or_not'] = all_features['YrBltAndRemod'].apply(lambda x: 1 if x != 0.0 else 0)
all_features['Total_Bathrooms'] = (all_features['FullBath'] 
                                   + (0.5 * all_features['HalfBath']) 
                                   + all_features['BsmtFullBath'] 
                                   + (0.5 * all_features['BsmtHalfBath']))

all_features['haspool'] = all_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_features['has2ndfloor'] = all_features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasgarage'] = all_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasbsmt'] = all_features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

# Comments:
# first, we calculated the total quality by simply get the average of two rating from our data: overall quality rating and overall condition rating.
# Then, the total area of each house is the sum of area for 1st floor, 2nd floor and basement

all_features.drop(['Utilities', 'Street', 'PoolQC', 'YearRemodAdd', 'FullBath', 'BsmtFullBath',
                   'HalfBath', 'BsmtHalfBath', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                   '3SsnPorch', 'ScreenPorch', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], 
                   axis=1, inplace=True)


log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                 'BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars',
                 'GarageArea', 'PoolArea','MiscVal', 'TotalSF']

for colname in log_features:
    all_features[colname+"_log"] = np.log1p(all_features[colname])

# make dummies for categorical variables
# Numerically encode categorical features because most models can only handle numerical features.
# pd.get_dummies automatically distinguish categorical variables from continuous.
all_features = pd.get_dummies(all_features).reset_index(drop=True)

# Just in case, remove any duplicated column names
all_features = all_features.loc[:, ~all_features.columns.duplicated()]

# train-test split
X = all_features.iloc[:len(trainLabels), :]
X_test = all_features.iloc[len(trainLabels):, :]
X.shape, trainLabels.shape, X_test.shape 



# models training part
# cv
k = KFold(n_splits=12, random_state=60, shuffle=True)
# Define error metrics
#def rmsle(y, y_pred):
#    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, x=X):
    rmse = np.sqrt(-cross_val_score(model, x, trainLabels, scoring="neg_mean_squared_error", cv=k))
    return rmse


# generate regressor 
xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=6000,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear',
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=6e-5,
                       #random_state=42,
                       n_jobs=16)

#train xgboost model
scores = {}
score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['xgb'] = (score.mean(), score.std())


# Ridge Regressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=k))

# train models

score = cv_rmse(ridge)
print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['ridge'] = (score.mean(), score.std())
scores['ridge']


# random forest regressor
rf = RandomForestRegressor(n_estimators=600,
                          max_depth=30,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          n_jobs=16,
                          random_state=42)

# train random forest models
score = cv_rmse(rf)
print("rf: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['rf'] = (score.mean(), score.std())


# Identify the best performing model
# Plot the predictions for each model
sns.set_style("white")
fig = plt.figure(figsize=(24, 12))

ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])
for i, score in enumerate(scores.values()):
    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')

plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)
plt.xlabel('Model', size=20, labelpad=12.5)
plt.tick_params(axis='x', labelsize=13.5)
plt.tick_params(axis='y', labelsize=12.5)

plt.title('Scores of Models', size=20)

plt.show()


#fit the best model, i.e., random forest
print('RandomForest')
rf_model_full_data = rf.fit(X, trainLabels)
# take a look at sample submission
submission = pd.read_csv(r"C:\Users\18612\Desktop\UMD_MSBA\758b\Projec\house-prices-advanced-regression-techniques\sample_submission.csv")
submission.shape

# concat prediction to submission
submission.iloc[:,1] = np.floor(np.expm1(rf_model_full_data.predict(X_test)))

# save submission to file
submission.to_csv("submission_regression.csv", index=False)