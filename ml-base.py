"""
Program Goes Over:   

- Data Wrangling: Numpy, Pandas | Apache Spark & Databricks
- ML Libaries: Scikit-Learn | Keras & TensorFlow
- Plotting: Matplotlib
"""

# Numpy is best for numerical data
# Numpy is memory efficient & indexing is very fast
# Numpy has better performace w/ 50k less rows

import numpy as np
from numpy.core.fromnumeric import transpose 

def numpyTest():

    # Example of a matrix
    A = np.array([[1, 4, 5], [-5, 8, 9]])   # 2x3
    B = np.array([[2,5,12], [4,3,29]])        # 2x3
    C = np.array([[5,2], [4,2], [8,7]])     # 3x2
    D = np.arange(15).reshape(3,5)         # 3x5

    # Example of a vector
    v = np.array([[8], [9], [1]])           # 3x1

    # Matrix Addition
    addResult = A + B

    # Matrix Multiplication
    dotResult = A.dot(C)
    dotResult2 = B.dot(D)

    # Matrix to Vector Multiplication
    dotResult3 = A.dot(v)

    # Matrix Transpose
    transposeResult = A.transpose()

    # Matrix Inversion

    print(addResult)
    print(dotResult)
    print(dotResult2)
    print(dotResult3)

    print(transposeResult)

    print(A.shape)
    print(C.shape)
    print(A[0].mean())  # Finds the mean of a set of features

# Pandas is best for Tabular Data
# Pandas consumes more memory and has slower indexing

import pandas as pd

class PandaTutorial():
    
    def __init__(self):
        # Create a 3x3 matrix
        grades = [  ['Faraz', 50.2, 29],
                    ['Rakin', 98.2, 28],
                    ['Jeff', 100.0, 22]]
 
        # Create a pandas dataframe of a 3x3 matrix
        df = pd.DataFrame(grades, columns=['Name', 'Grade', 'Age'])

        # Addding an additional column to frame
        df["Status"] = pd.Categorical(["test", "train", "test"])

        self.df = df

        # Importing CSV into a dataframe
        #df = pd.read_csv('./sampleData/housing.csv')

    def pandaDetails(self):
        df = self.df

        # View top of frame
        print(df.head())

        # Gives the shape of the data  
        print(df.shape)

        # View bottom of frame
        print(df.tail())

        # View Index
        print(df.index)

        # View Stats on frame
        print(df.describe())

    def pandaInfo(self):
        df = self.df

        # Select a specfic data point (row, column)
        print(df.iloc[1]['Name'])

        #Select a specfic column
        print(df["Status"])

        # Filtering w/index
        print(f' Filter by index: {df[df["Age"] < 26]}')

        # Filtering by criteria
        print(f' Filter by criteria: {df[df["Status"].isin(["test"])]}')

        # Find what data is null
        print(f' Finding Null Data: {df[df.isnull().any(axis=1)]}')

        # Find what data is null
        print(f' Finding How Many Pieces of Data are Null: {df.isnull().sum()}')

        # Find if data is correlated
        print(f' Seeing if columns are correlated: {df["Age"].corr(df["Grade"])}')

    def pandaManipulation(self, col):
        df = self.df

        # Sorting by Values
        print(df.sort_values(by=col))

        # Applying functions to data
        #print(df.apply(lambda x: x+2))

        # Transposing Frame
        print(df.T)

        # Drop rows w/ missing data
        df.dropna(how="any")

        # Fill missing data
        #df.fillna(value=5)

# Matplotlib is great for plotting data

from matplotlib import pyplot as plt

class Plotting(): 
    
    def __init__(self, filePath):
        df = pd.read_csv(filePath)
        
        # Remove any rows with missing data
        self.df = df.dropna(axis=0, how='any')
        print(self.df.head())

    def graphExamples(self, col1, col2):
        df = self.df

        # Create a figure for 2 subplots (2 rows, 1 column)
        fig, ax = plt.subplots(2, 1, figsize = (10,4))

        # Create a scatter plot
        ax[0].scatter(df[col1], df[col2], color ="green") 
        
        # Create a bar graph
        ax[1].bar(x=df[col1], height=df[col2])

        fig.suptitle('Key Data')
        plt.show()
        
    def showDistribution(self, col):
        df = self.df

        # Calculate Mean, Median, Mode, Min, Max
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        med_val = df[col].median()
        mod_val = df[col].mode()[0]

        # Create a figure for 2 subplots (2 rows, 1 column)
        fig, ax = plt.subplots(2, 1, figsize = (10,4))

        # Create a histogram
        ax[0].hist(df[col])
        ax[0].set_ylabel('Frequency')
        
        # Add lines for the statistics
        ax[0].axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
        ax[0].axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
        ax[0].axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
        ax[0].axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
        ax[0].axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

        # Create a box and whiskers plot
        ax[1].boxplot(df[col], vert=False)
        ax[1].set_xlabel('Value')

        plt.show()

# Ski-Kit Learn is a rich library for deep and gritty machine learning

# Sci-Kit Regression Algorithms
from sklearn.linear_model import LinearRegression, Lasso 
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# Sci-Kit Classification Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# See https://docs.microsoft.com/en-us/learn/modules/train-evaluate-classification-models/3-exercise-model 

# For normalizing and/or tranformating letter data into categorical data ex. 0,1,2...
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
# See https://docs.microsoft.com/en-us/learn/modules/train-evaluate-regression-models/7-exercise-optimize-save-models for Pipeline

# Regression Metrics
from sklearn.metrics import mean_squared_error, make_scorer, r2_score 

# Classifiction Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

# For Randomly and Easily Splitting Data for Testing & Training
from sklearn.model_selection import train_test_split

class MLModels():
    def __init__(self, filePath):
        df = pd.read_csv(filePath)
        
        # Remove any rows with missing data
        self.df = df.dropna(axis=0, how='any')
        print(self.df.head())

        self.model = ''
        self.predictions = []
        self.actual_test_values = []

    def dropColumns(self, col):
        self.df = self.df.drop([col], axis=1)
        print(self.df.head())

    def normalizeData(self, isManual, col1="", col2=""):
        df = self.df

        if isManual:
            # Normalize Data Manually
            self.df = (df-df.mean())/df.std()
        else:
            # Normalize two columns using scikit learn so that values fall between -1 - 1
            scaler = MinMaxScaler()
            df[[col1,col2]] = scaler.fit_transform(df[[col1 , col2]])
            self.df = df

            print(f' Finding if columns are correlated: {df[col1].corr(df[col2])}')

        print(self.df.head())

    def generateLinearModel(self, modelType, feature, label):
        if modelType == "hyperParamGradient":
            self.performHyperParamExample(feature, label)
        else:
            x = self.df[feature].values.reshape((-1, 1))
            y = self.df[label]

            if modelType == 'Linear':
                self.model = LinearRegression()
                self.performRegression(x, y)
            elif modelType == "Lasso":
                self.model = Lasso()
                self.performRegression(x, y, "lasso")
            elif modelType == "Decision-Tree":
                self.model = DecisionTreeRegressor()
                self.performRegression(x, y, "tree")
            elif modelType == "Ensemble":
                self.model = RandomForestRegressor()
                self.performRegression(x, y, "other")
            elif modelType == "GradientBoost":
                self.model = GradientBoostingRegressor()
                self.performRegression(x, y, "other")

            self.plotModel(x, y)

        return self.model

    def performRegression(self, x, y, special=None):
        # Splits data into 70% training and 30% test data to check predictions
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

        # Alternative and more accurate method for splitting test data
        altMethod = """
            kfold = StratifiedKFold(n_splits=10, random_state=7, shuffle=True)
            for train_index, test_index in kfold.split(X, y):
                print("TRAIN:", train_index, "TEST:", test_index)
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
            """

        # Learning Algorithm
        self.model.fit(x_train, y_train)                        # Calculates the optimal values using the data for the linear function

        findIntercept = False
        if special:
            if special == 'tree':
                # Visualize the model tree
                tree = export_text(self.model)
                print(tree)
        else:
            findIntercept = True

        if findIntercept:
            # ML Model Info
            x1 = self.model.intercept_                              # Finds the intercept 5.63333
            x0 = self.model.coef_                                   # Finds the slope of the graph Ex. [.54]
            print(f'Slope: {x0}\nIntercept: {x1}')

        # Manual Hypothesis Function of predicting some data
        #self.predictions = self.model.intercept_ + self.model.coef_ * x

        # Faster Built in Function used to Predict Data
        self.predictions = self.model.predict(x_test)
        self.actual_test_values = y_test

        print(f'Predicted data is: {self.predictions}')
        print(f'Actual data is: {y_test}')

        # Finds the cost error of the ML Algorithm
        r_sq = self.model.score(x_test, y_test)                     # Find the coefficiant of determination
        msq_err = mean_squared_error(y_test, self.predictions)      # Finds the Mean Square Error
        rmse = np.sqrt(msq_err)

        print(f'Root Mean Square Error: {rmse}\nCoeff. Of Determination: {r_sq}')

    def performHyperParamExample(self, feature_x_col, predict_y_col):
        x = self.df[feature_x_col].values.reshape((-1, 1))
        y = self.df[predict_y_col]

        # Splits data into 70% training and 30% test data to check predictions
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

        # Use a Gradient Boosting algorithm
        alg = GradientBoostingRegressor()

        # Try these hyperparameter values
        params = {
        'learning_rate': [0.1, 0.5, 1.0],
        'n_estimators' : [50, 100, 150]
        }

        # Find the best hyperparameter combination to optimize the R2 metric
        score = make_scorer(r2_score)
        gridsearch = GridSearchCV(alg, params, scoring=score, cv=3, return_train_score=True)
        gridsearch.fit(x_train, y_train)
        print("Best parameter combination:", gridsearch.best_params_, "\n")

        # Get the best model
        self.model = gridsearch.best_estimator_
        print(self.model, "\n")

        # Evaluate the model using the test data
        predictions = self.model.predict(x_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        print(f'Root Mean Square Error: {rmse}\nCoeff. Of Determination: {r2}')

        self.plotModel(x, y)

        return self.model

    def generateClassifierModel(self, modelType, features, label):
        x, y = self.df[features].values, self.df[label].values 
        for n in range(0,4):
            print("Patient", str(n+1), "\n  Features:",list(x[n]), "\n  Label:", y[n])

        for col in features:
            self.df.boxplot(column=col, by='diagnosis', figsize=(6,6))
            plt.title(col)
        plt.show()

        # Splits data into 70% training and 30% test data to check predictions
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

        # Alternative and more accurate method for splitting test data
        altMethod = """
            kfold = StratifiedKFold(n_splits=10, random_state=7, shuffle=True)
            for train_index, test_index in kfold.split(X, y):
                print("TRAIN:", train_index, "TEST:", test_index)
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
            """

        # Pipelines can be added in order for the columns that have a string value to be converted to numerical data
        numeric_features = [0]
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

        # Encode the String columns
        categorical_features = [1]
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), 
                                                    ('cat', categorical_transformer, categorical_features)])
                

        # Train a logistic regression model on the training set
        if modelType == "RandomForest":
            self.model = RandomForestClassifier(n_estimators=100)
        elif modeType == "Regression":
            # Set regularization rate
            reg = 0.01
            self.model = LogisticRegression(C=1/reg, solver="liblinear")

        self.model = Pipeline(steps=[('preprocessor', preprocessor), ('logregressor', self.model)])
        self.model.fit(x_train, y_train)

        # Predict Data
        self.predictions = self.model.predict(x_test)
        self.actual_test_values = y_test

        print(f'Predicted data is: {self.predictions}')
        print(f'Actual data is: {self.actual_test_values}')

        # Obtain Metrics to verify Accuracy Of Model
        cm = confusion_matrix(self.actual_test_values, self.predictions)
        accuracy = accuracy_score(self.actual_test_values,  self.predictions)
        precision = precision_score(self.actual_test_values, self.predictions, average=None)
        recall = recall_score(self.actual_test_values, self.predictions, average=None)
        
        stats = { 
            'confusion_matrix': cm,
            'accuracy': accuracy,
            "precision":precision,
            "recall":recall
        }

        print(stats)

        return self.model

    def plotModel(self, x, y):

        # Plotting Against All Data
        #plt.scatter(x, y, color ="green")                           # for scatter graph of training data
        #plt.plot(x, self.predictions, color ="red")                 # for line graph of regression line

        # Comparing Predictions to Actual Values
        plt.scatter(self.actual_test_values, self.predictions)
        plt.xlabel('Actual Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Predicted House Values')

        # Overlay the Regression Line
        z = np.polyfit(self.actual_test_values, self.predictions, 1)
        p = np.poly1d(z)
        plt.plot(self.actual_test_values, p(self.actual_test_values), color='magenta')
        
        plt.show()

# ---------------------------------------------------


def numpy_pandas_plotlibTest():
    #numpyTest
    
    pandaExample = PandaTutorial()
    #pandaExample.pandaDetails()
    #pandaExample.pandaInfo()
    #pandaExample.pandaManipulation('median_house_value')

    pltExample = Plotting('./sampleData/housing.csv')
    pltExample.graphExamples('median_income', 'median_house_value')
    pltExample.showDistribution('median_house_value')

def exportMLModel(model, type):
    import joblib

    # Save the model as a pickle file
    filename = f'./ml_model_example_{type}.pkl'
    joblib.dump(model, filename)

def testRegression():
    mlExample = MLModels('./sampleData/housing.csv')
    mlExample.dropColumns('ocean_proximity')
    mlExample.normalizeData(False, 'median_income', 'median_house_value')
    
    mlExample.generateLinearModel('Linear', 'median_income', 'median_house_value')
    #mlExample.generateLinearModel('Lasso', 'median_income', 'median_house_value')
    #mlExample.generateLinearModel('Decision-Tree', 'median_income', 'median_house_value')
    #mlExample.generateLinearModel('Ensemble', 'median_income', 'median_house_value')
    #mlExample.generateLinearModel('GradientBoost', 'median_income', 'median_house_value')

    model = mlExample.generateLinearModel('hyperParamGradient', 'median_income', 'median_house_value')
    exportMLModel(model, 'Regression')

def testClassification():
    mlExample = MLModels('./sampleData/Breast_cancer_data.csv')
    features = [ 'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']
    label = 'diagnosis'
    
    #model = mlExample.generateClassifierModel('Logistic', features, label)
    model = mlExample.generateClassifierModel('RandomForest', features, label)
    exportMLModel(model, 'Classification')

if __name__ == '__main__':
    #numpy_pandas_plotlibTest()
    #testRegression()
    testClassification()