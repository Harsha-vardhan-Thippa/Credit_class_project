import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sys
import os
import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging("Random_sample_imputation.py")

def Random_sample(x_train,x_test):
    try:
        logger.info('======================================Before Random_sample=====================================')
        logger.info(f"Training data shape: {x_train.shape}")
        logger.info(f"Test data shape: {x_test.shape}")
        logger.info(f'{x_train.isnull().sum()}')
        logger.info(f'{x_test.isnull().sum()}')

        for i in x_train.columns:
            if x_train[i].isnull().sum()>0 and x_test[i].isnull().sum()>0:
                logger.info(f' Random sample imputation is performed on {i}')

                non_missing_values_train = x_train[i].dropna()
                x_train.loc[x_train[i].isnull(),i] = np.random.choice(a=non_missing_values_train, size=x_train[i].isnull().sum())

                non_missing_values_test = x_test[i].dropna()
                x_test.loc[x_test[i].isnull(), i] = np.random.choice(a=non_missing_values_test,
                                                                       size=x_test[i].isnull().sum())
        logger.info('======================================After Random_sample=====================================')
        logger.info(f"Training labels shape: {x_train.shape}")
        logger.info(f"Test labels shape: {x_test.shape}")
        logger.info(f'{x_train.isnull().sum()}')
        logger.info(f'{x_test.isnull().sum()}')


    except Exception as e:
        logger.error(e)
        logger.info("Failed")

    return x_train, x_test