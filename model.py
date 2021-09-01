import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV, Lasso, ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
import pickle
from logger import Logger


class Model:
    def __init__(self, file):
        self.df = pd.read_csv(file).iloc[:, 2:]

    def independent_features(self):
        try:
            return self.df.loc[:, [col for col in self.df.columns if col != 'Air temperature [K]']]
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Unable to create independent features from the dataframe \n {str(e)}')

    def dependent_feature(self):
        try:
            return self.df.loc[:, 'Air temperature [K]']
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Unable to create dependent feature from the dataframe \n {str(e)}')

    def missing_imputation(self):
        try:
            a = self.df.describe().loc['count'] < 10000
            if len(a[a.values == True].index) > 0:
                for col in a[a.values == True].index:
                    if col == 'Air temperature [K]':
                        continue
                    elif col in ['Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']:
                        self.df.col.fillna(self.df.col.mean(), inplace=True)
                    else:
                        self.df.col.fillna(self.df.col.mode()[0], inplace=True)
            else:
                Logger('test.log').logger('INFO', f'No missing values')
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Error imputing missing values \n {str(e)}')

    def categorical_label_encoding(self, col):
        try:
            le = LabelEncoder()
            le.fit(self.df[col])
            filename = f'{col}_label_encoder.sav'
            pickle.dump(le, open(filename, 'wb'))
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Couldnt label encode categorical variable \n {str(e)}')

    def one_hot_encoder(self, col):
        try:
            if self.df.dtypes[col] == object:
                self.categorical_label_encoding(col)
                oe = OneHotEncoder()
                oe.fit(self.df[[col]])
                filename = f'{col}_one_hot_encoder.sav'
                pickle.dump(oe, open(filename, 'wb'))
            else:
                oe = OneHotEncoder()
                oe.fit(self.df[[col]])
                filename = 'one_hot_encoder.sav'
                pickle.dump(oe, open(filename, 'wb'))
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Couldnt one hot encode categorical variable \n {str(e)}')

    def std_scaler(self, x_numerical_values):
        try:
            x = self.df[x_numerical_values]
            scaler = StandardScaler()
            scaler.fit(x)
            filename = 'std_scaler.sav'
            pickle.dump(scaler, open(filename, 'wb'))
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Numerical columns couldnt be scaled \n {str(e)}')

    def data_transformation(self):
        try:
            self.missing_imputation()
            x = self.independent_features()
            y = self.dependent_feature()
            se = pickle.load(open("std_scaler.sav", 'rb'))
            #le = pickle.load(open("Type_label_encoder.sav", 'rb'))
            #oe = pickle.load(open('Type_one_hot_encoder.sav', 'rb'))
            x[['Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']] = se.transform(x[['Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']])
            categorical_type = pd.get_dummies(x['Type'])
            x = pd.concat([x, categorical_type], axis=1)
            x.drop(columns='Type', inplace=True)
            return x, y
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Couldnt transform the dataset \n {str(e)}')

    def train_test_split(self):
        try:
            x, y = self.data_transformation()
            x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.2, random_state=42)
            return x_train, x_test, y_train, y_test
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Couldnt split the dataset into train test sets \n {str(e)}')

    def adj_r2_score(self, model, x, y):
        try:
            r2 = model.score(x, y)
            n = x.shape[0]
            p = x.shape[1]
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            return adjusted_r2
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Error scoring adj r2 \n {str(e)}')

    def lr_model(self):
        try:
            x_train, x_test, y_train, y_test = self.train_test_split()
            lr = LinearRegression()
            lr.fit(x_train, y_train)
            return 'linear', lr, self.adj_r2_score(lr, x_test, y_test)
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Error creating a LR model \n {str(e)}')

    def lasso_lr_model(self):
        try:
            x_train, x_test, y_train, y_test = self.train_test_split()
            lassocv = LassoCV(alphas=None, cv=50, max_iter=200000, normalize=True)
            lassocv.fit(x_train, y_train)
            lasso = Lasso(alpha=lassocv.alpha_)
            lasso.fit(x_train, y_train)
            return 'lasso', lasso, self.adj_r2_score(lasso, x_test, y_test)
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Error creating a Lasso model \n {str(e)}')

    def ridge_lr_model(self):
        try:
            x_train, x_test, y_train, y_test = self.train_test_split()
            ridgecv = RidgeCV(alphas=np.random.uniform(0, 10, 50), cv=10, normalize=True)
            ridgecv.fit(x_train, y_train)
            ridge_lr = Ridge(alpha=ridgecv.alpha_)
            ridge_lr.fit(x_train, y_train)
            return 'ridge', ridge_lr, self.adj_r2_score(ridge_lr, x_test, y_test)
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Error creating a ridge model \n {str(e)}')


    def elastic_lr_model(self):
        try:
            x_train, x_test, y_train, y_test = self.train_test_split()
            elastic = ElasticNetCV(alphas=None, cv=10)
            elastic.fit(x_train, y_train)
            elastic_lr = ElasticNet(alpha=elastic.alpha_, l1_ratio=elastic.l1_ratio_)
            elastic_lr.fit(x_train, y_train)
            return 'elastic', elastic_lr, self.adj_r2_score(elastic_lr, x_test, y_test)
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Error creating a elastic model \n {str(e)}')

    def save_model(self, model):
        try:
            filename = 'model.sav'
            pickle.dump(model, open(filename, 'wb'))
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Error saving the LR model \n {str(e)}')

    def best_model(self):
        try:
            models = {}
            best_model = None
            for i in [self.lr_model(), self.lasso_lr_model(), self.ridge_lr_model(), self.elastic_lr_model()]:
                models[i[0]] = [i[1], i[2]]
            best_model = sorted(models.values(), key=lambda val: val[1], reverse=True)[0][0]
            filename = 'model.sav'
            pickle.dump(best_model, open(filename, 'wb'))
            return best_model
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Couldnt save the best model \n {str(e)}')
