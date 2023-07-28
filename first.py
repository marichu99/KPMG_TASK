import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import scipy as sp
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.svm import SVR
from tensorflow import keras
from keras import layers



df_NewCustomer=pd.read_excel("KPMG_VI_New_raw_data_update_final.xlsx",sheet_name="NewCustomerList",skiprows=1)

# visualize the dataframe
print(df_NewCustomer.head())

# check all column names
for col in df_NewCustomer.columns:
    print(col)
    if("Unnamed" in str(col)):
        # drop the column
        df_NewCustomer.drop(col,axis=1,inplace=True)

df_numerical=df_NewCustomer.select_dtypes(exclude=["object","datetime64[ns]"])

print("the numerical data before normalization")
print(df_numerical)
# numerical independent
df_IndependentNumerical=df_numerical.drop("past_3_years_bike_related_purchases",axis=1)
print(df_IndependentNumerical)
# since we are trying to see which of the new customers should be targeted mostly for marketting lets correlate some of the predictor variables with the target variable of bike sales in the past 3 years
df_Independent=df_NewCustomer.drop("past_3_years_bike_related_purchases",axis=1)
df_Target=df_NewCustomer["past_3_years_bike_related_purchases"]

# try binning the target column
# change the datatype of the target variable into number
df_Target.astype(dtype="Int64")
print(f"The target variable is of datatype {df_Target.dtype}")
# lets start with numerical and continuous data
# normalize
col_transformer= make_column_transformer(
    (StandardScaler(),df_IndependentNumerical.columns)
)
df_numeric=col_transformer.fit_transform(df_IndependentNumerical)
df_numeric=pd.DataFrame(df_numeric)
df_numeric.columns=df_IndependentNumerical.columns
print("the numerical data")
print(df_numeric)
print(df_Target)

input_shape=[len(df_Independent.columns)]




# for col in df_Independent.columns:
#     if (col in df_numeric.columns):
#         corr,pval=sp.stats.pearsonr(df_Independent[col],df_Target)
#         # normalize the predictor variables first to see whether you can get different results
#         print(f" The {col} column has {df_Independent[col].dtype} and correlation factor of {corr}")

# sns.heatmap(df_numerical.corr())
# plt.show()


# """
#  get the density distribution of bike related purchases in the past three years variable 
#  to get a visual perception on how many bikes were bought in the past three years

# """
# sns.kdeplot(data=df_NewCustomer,x="past_3_years_bike_related_purchases")
# # show the statistical summary of the target variable
# sns.violinplot(data=df_NewCustomer,x="past_3_years_bike_related_purchases")
# plt.show()

# split the data
x_train,x_valid,y_train,y_valid=train_test_split(df_IndependentNumerical,df_Target,test_size=0.25,train_size=0.75,random_state=0)

y_train=np.array(y_train)
y_train=y_train.reshape(-1,1)
s_valYScaler=StandardScaler()
print(x_train)
s_calXScaler=StandardScaler()
x_train=s_calXScaler.fit_transform(x_train)
y_train=s_valYScaler.fit_transform(y_train)
s_vector=SVR(kernel="rbf")
lr=LinearRegression()
hxgb=HistGradientBoostingRegressor(loss="absolute_error")

# train all the models
s_vector.fit(x_train,y_train)
lr.fit(x_train,y_train)
hxgb.fit(x_train,y_train)

dl_model=keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(128,activation="relu",input_shape=input_shape),
    layers.Dropout(0.3),
    layers.Dense(128,activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64,activation="selu"),
    layers.Dense(1)
])

dl_model.compile(
    optimizer="adam",
    loss="mae"
)

metric_history=dl_model.fit(x=x_train,y=y_train,batch_size=256,epochs=20,validation_data=[x_valid,y_valid])

dl_metric=pd.DataFrame(metric_history.history)

# get the last error
dl_last_error=dl_metric["loss"].iloc[-1]

# evaluate the performance of the various models
s_predictions=s_vector.predict(x_valid)
s_metric=mae(y_valid,s_predictions)
lr_predictions=lr.predict(x_valid)
l_metric=mae(y_valid,lr_predictions)
hxgb_predictions=hxgb.predict(x_valid)
hxgb_metric=mae(y_valid,hxgb_predictions)

print(f"The SVM has {s_metric} MAE")
print(f"The LR has {l_metric} MAE")
print(f"The HXGB has {hxgb_metric} MAE")
print(f"The Keras DL has {dl_last_error} MAE")