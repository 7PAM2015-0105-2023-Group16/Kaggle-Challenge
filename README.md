# Kaggle challenge
This Kaggle-Challenge build a predictive model to determine which passenger from a hypothetical spaceship were transported to an alternate dimension based on spaceship titanic dataset.




IMPORTED MODULES AND LIBRARIES

import numpy as np

import pandas as pd

import missingno as msno

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

from google.colab import files




In our first attempt we got a point 0.Then in second attempts we used Random forest and we got a point 0.78816.


In order to get higher value we tried in our 3rd attempt Gradiant booster model that gave us the value 0.79611 which is much better than previous model.

Used Explainable AI(Permutation Importance) methods to find out feature importance.

We did preprocessing steps and set PCA with number of components=2 and thereafter trained the model again and finally got a better accuracy for predicted values which is 0.79635. This is our best score so far.



