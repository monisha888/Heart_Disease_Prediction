"""
Load the CSV dataset and create the train/test split.
Transform/handle outliers in X_train and X_test via an external function.
Select the most relevant feature columns via an external function.
Balance training data with SMOTE, then scale features with StandardScaler.
Train and compare multiple models on the balanced, scaled data.
Train a Gaussian Naive Bayes model, evaluate it, and save model + scaler as .pkl files.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import pickle
import warnings
warnings.filterwarnings("ignore")
from log import setup_logging
logger = setup_logging("main")
from sklearn.model_selection import train_test_split
from variable_transformation import variable_transformation_outliers
from feature_selection import best_col
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from all_models import common
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
class HEART:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            logger.info(f"NUll values in the values  : {self.df.isnull().sum()}")
            self.X = self.df.iloc[: , :-1]
            self.y = self.df.iloc[: , -1]
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X , self.y,test_size=0.2 , random_state=42)
            logger.info(f"Training dataset size : \n {self.X_train.shape}  => {self.y_train.shape}")
            logger.info(f"Testing dataset size : \n {self.X_test.shape} => {self.y_test.shape}")
        except Exception as e:
            er_type,er_msg,er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")
    def variable_transformation_outliers(self):
        try:
            self.X_train,self.X_test=variable_transformation_outliers(self.X_train,self.X_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")
    def best_col(self):
        try:
            self.X_train, self.X_test = best_col(self.X_train, self.X_test,self.y_train,self.y_test)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")
    def data_balancing(self):
        try:
            logger.info(f"Data balancing : \n {self.X_train.shape} => {self.y_train.shape}")
            logger.info(f"no of rows in target:{1}:{sum(self.y_train==1)}")
            logger.info(f"no of columns in target:{0}:{sum(self.y_train == 0)}")
            sm_obj=SMOTE(random_state=42)
            self.X_train_bal,self.y_train_bal=sm_obj.fit_resample(self.X_train,self.y_train)
            logger.info(f"After Data balancing:\n{self.X_train.shape}=>{self.y_train_bal.shape}")
            logger.info(f"no of rows in target:{1}:{sum(self.y_train_bal==1)}")
            logger.info(f"no of columns in target:{0}:{sum(self.y_train_bal==0)}")
            #Scale down using Z score BIG VALUES TO SMALL USING Z SCORE
            self.sc=StandardScaler()
            self.X_train_bal_scaled=self.sc.fit_transform(self.X_train_bal)
            self.X_test_scaled=self.sc.fit_transform(self.X_test)
            #training variables self.X_train_bal_scaled,self.y_train_bal
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")
    def all_model(self):
        try:
            common(self.X_train_bal_scaled,self.y_train_bal,self.X_test_scaled,self.y_test)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

    def best_model(self):
        try:
            nb_obj = GaussianNB(var_smoothing=np.float64(1e-12))
            nb_obj.fit(self.X_train_bal_scaled, self.y_train_bal)
            logger.info(f"accuracy score:{accuracy_score(self.y_test, nb_obj.predict(self.X_test_scaled))}")
            logger.info(f"confusion matrix:{confusion_matrix(self.y_test, nb_obj.predict(self.X_test_scaled))}")
            logger.info(
                f"classification report:{classification_report(self.y_test, nb_obj.predict(self.X_test_scaled))}")
            # hyperparamter tuning
            # parameter_list={
            #     "var_smoothing":np.logspace(-12,-1,12)
            # }
            # grid_obj=GridSearchCV(estimator=nb_obj,param_grid=parameter_list,cv=10,scoring="accuracy",n_jobs=-1)
            # grid_obj.fit(self.X_train_bal_scaled,self.y_train_bal)
            # logger.info(f"best parameters:{grid_obj.best_params_}")
            # logger.info(f"best score:{grid_obj.best_score_}")
            predict = np.array([[63, 1, 3, 150, 2.3, 0, 1]])
            self.sc.transform(predict)
            logger.info(nb_obj.predict(predict)[0])
            with open('model.pkl','wb') as f:
                pickle.dump(nb_obj,f)
            with open('scaled.pkl','wb') as p:
                pickle.dump(self.sc,p)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")


if __name__ == "__main__":
    try:
        obj = HEART("heart.csv")
        obj.variable_transformation_outliers()
        obj.best_col()
        obj.data_balancing()
        obj.all_model()
        obj.best_model()

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")