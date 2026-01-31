'''in this we are going to perform the code cleaning'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sys
import os
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging("main.py")

from Random_sample_imputation import Random_sample
from variable_T_outliers import variable_transformation_outliers
from feature_selection import complete_feature_selection
from c_to_n import categorical_to_numeric
from imbalance_data import balancing_data


class CREDIT_CARD_DATA:
    def __init__(self, path):
        try:
            self.path = path
            self.df = pd.read_csv(path)
            logger.info("Data loaded successfully ")
            logger.info("-----------------------------------Before Data cleaning--------------------------------")
            logger.info(f'{self.df.sample(10)}')
            logger.info(f'Total Rows in the data is {self.df.shape[0]}')
            logger.info(f'Total Columns in the data is {self.df.shape[1]}')
            logger.info(f'{self.df.sample(10)}')
            logger.info("-----------------------------------Null Values information------------------------------")
            logger.info(f'{self.df.isnull().sum()}')
            logger.info(f'After {self.df.tail(5)}')
            logger.info("----------------------------------After data cleaning------------------------------------")
            self.df.drop([150000,150001],axis=0,inplace=True)
            self.df.drop("MonthlyIncome.1",axis=1,inplace=True)
            logger.info(f'{self.df.isnull().sum()}')
            logger.info(f'After {self.df.tail(5)}')
            logger.info("==========================================================================================")
            for i in self.df.columns:
                if self.df[i].isnull().sum() > 0 and self.df[i].dtype == "object":
                    self.df[i] = pd.to_numeric(self.df[i])
                logger.info(f'{i} ---> {self.df[i].dtype}')
            logger.info("==========================================================================================")
            self.x = self.df.iloc[:,:-1]
            self.y = self.df.iloc[:,-1]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,self.y,test_size=0.2,random_state=42)


        except Exception as e:
            logger.info("Data loading failed")
            logger.error(e)

    def Handling_missing_values(self):

        try:
            logger.info('====================================== Before Handling_missing_values ====================== =====================================')
            logger.info(f"Training data shape: {self.x_train.shape}")
            logger.info(f"Test data shape: {self.x_test.shape}")
            logger.info(f'{self.x_train.isnull().sum()}')
            logger.info(f'{self.x_test.isnull().sum()}')
            self.x_train,self.x_test = Random_sample(self.x_train,self.x_test)
            logger.info('====================================== After Handling_missing_values ====================== =====================================')
            logger.info(f"Training labels shape: {self.x_train.shape}")
            logger.info(f"Test labels shape: {self.x_test.shape}")
            logger.info(f'{self.x_train.isnull().sum()}')
            logger.info(f'{self.x_test.isnull().sum()}')

        except Exception as e:
            logger.info("Data loading failed")
            logger.error(e)

    def Variable_transformation(self):
        try:
            logger.info(f"{self.x_train.columns}")
            logger.info(f"{self.x_test.columns}")
            logger.info("--------------------------------------------------------------")
            self.x_train_num = self.x_train.select_dtypes(exclude="object")
            self.x_train_cat = self.x_train.select_dtypes(include="object")
            self.x_test_num = self.x_test.select_dtypes(exclude="object")
            self.x_test_cat = self.x_test.select_dtypes(include="object")
            logger.info(f'{self.x_train_num.columns}')
            logger.info(f'{self.x_train_cat.columns}')
            logger.info(f'{self.x_test_num.columns}')
            logger.info(f'{self.x_test_cat.columns}')
            logger.info(f"{self.x_train_num.shape}")
            logger.info(f"{self.x_train_cat.shape}")
            logger.info(f"{self.x_test_num.shape}")
            logger.info(f"{self.x_test_cat.shape}")
            self.x_train_num, self.x_test_num = variable_transformation_outliers(self.x_train_num,self.x_test_num)
            logger.info("After variable transformation")
            logger.info(f"{self.x_train_num.columns}---->{self.x_train_num.shape}")
            logger.info(f"{self.x_test_num.columns}---->{self.x_test_num.shape}")

        except Exception as e:
            logger.info("Data loading failed")
            logger.error(e)

    def Feature_selection(self):
        try:
            logger.info(f"{self.x_train.info}")
            logger.info(f"Before Feature selection (train): {self.x_train_num.columns}---->{self.x_train_num.shape}")
            logger.info(f" Before Feature selection (test): {self.x_test_num.columns}---->{self.x_test_num.shape}")
            self.x_train_num,self.x_test_num = complete_feature_selection(self.x_train_num,self.x_test_num,self.y_train)
            logger.info(f"After Feature selection (train): {self.x_train_num.columns}---->{self.x_train_num.shape}")
            logger.info(f"After Feature selection (test): {self.x_test_num.columns}---->{self.x_test_num.shape}")
        except Exception as e:
            logger.info("Data loading failed")
            logger.error(e)

    def cat_to_num(self):
        try:
            for i in self.x_train_cat.columns:
                logger.info(f'{i}---->{self.x_train_cat[i].unique()}')
            logger.info(f" before converting categorical to numerical : {self.x_train_cat.columns}")
            logger.info(f" before converting categorical to numerical : {self.x_test_cat.columns}")

            self.x_train_cat,self.x_test_cat = categorical_to_numeric(self.x_train_cat,self.x_test_cat)

            logger.info(f" after converting categorical to numerical : {self.x_train_cat.columns}")
            logger.info(f" after converting categorical to numerical : {self.x_test_cat.columns}")

            logger.info(f"{self.x_train_cat.shape}")
            logger.info(f"{self.x_test_cat.shape}")
            logger.info(f"{self.x_train_cat.isnull().sum()}")
            logger.info(f"{self.x_test_cat.isnull().sum()}")

            logger.info(f"===========================================================")
            self.x_train_num = self.x_train_num.reset_index(drop=True)
            self.x_test_num = self.x_test_num.reset_index(drop=True)

            self.x_train_cat = self.x_train_cat.reset_index(drop=True)
            self.x_test_cat = self.x_test_cat.reset_index(drop=True)

            self.x_training_data = pd.concat([self.x_train_num, self.x_train_cat],axis=1)
            self.x_testing_data = pd.concat([self.x_test_num, self.x_test_cat],axis=1)

            logger.info(f"After combining the training data : \n {self.x_training_data.columns}")
            logger.info(f"After combining the testing data : \n {self.x_testing_data.columns}")

            logger.info(f' shape of the training data : {self.x_training_data.shape}')
            logger.info(f' shape of the testing data : {self.x_testing_data.shape}')

            logger.info(f'How many null values are in training data : \n {self.x_training_data.isnull().sum()}')
            logger.info(f'How many null values are in testing data : \n {self.x_testing_data.isnull().sum()}')

            logger.info(f"sample data : \n {self.x_training_data.sample(10)}")

        except Exception as e:
            logger.info("Data loading failed")
            logger.error(e)

    def balance_the_data(self):
        try:
            self.y_train = self.y_train.map({"Good":1, "Bad":0}).astype(int)
            self.y_test = self.y_test.map({"Good":1, "Bad":0}).astype(int)
            balancing_data(self.x_training_data,self.y_train,self.x_testing_data,self.y_test)

        except Exception as e:
            logger.info("Data loading failed")
            logger.error(e)










if __name__ == "__main__":
    try:
        obj = CREDIT_CARD_DATA(r"C:\Users\harsh\OneDrive\Desktop\ml_project\ml_pipeline\creditcard.csv")
        obj.Handling_missing_values()
        obj.Variable_transformation()
        obj.Feature_selection()
        obj.cat_to_num()
        obj.balance_the_data()

    except FileNotFoundError as e:
        logger.info("Data loading failed")
        logger.error(e)
