# SC1015 DataScience-Proj

### Done by: 
1. Muhammad Nabil Hakeem (U2122831K)
2. Su Gaoyang (U2121723E)
3. Scott Wong  (U2122366G)

## Background:


##

## Please view the codes in this order:
1. [Data Extraction And Data Preparation](https://github.com/Note06/DataScience-Proj/blob/main/1015-Proj%20EDA.ipynb)
2. [Using Machine Learning to get feature importance](https://github.com/Note06/DataScience-Proj/blob/main/All%20variable%20machine%20learning.ipynb)
3. [Prediction Models](https://github.com/Note06/DataScience-Proj/blob/main/Prediction%20using%20Feature%20Importance.ipynb)


## Problem Statement: 
With so many different factors that can affect water quality, which ones have the most impact?<br>
We decide to use the idea of feature importance. By identifying data that impacts the decision the most we can increase our data accuracy prediction and improve efficiency. 

## Dataset from:
MsSmartyPants. (2021, June 30). Water quality. Kaggle. Retrieved February 3, 2022, from https://www.kaggle.com/datasets/mssmartypants/water-quality

## Models Used:
1. Random Forest (In presentation, used for finding feature importance, and Prediction of the model)
2. Neural Network (In presentation, used for finding feature importance)
3. K Nearest Neighbours (Never discuss in the slide, but performed the model)
4. Logistic Regression (Never discuss in the slide, but performed the model)
5. Linear Discriminant Analysis & Principle Discriminant Analysis (Used for prediction of the model 

## Conclusion:
1. The 2 variables that appear the most number of times are Aluminium and Cadmium. From this, we can tell that the 2 variables are key to ensuring that the water is safe to drink.
2. Using ANN to get the feature importance yields good results under random forest prediction, but not as optimal for LDA. And furthermore, ANN doesn't tell you the process of prediction. So for certain predictions could not be explained and described scientifically. Therefore, resorting to other models, through examining Correlation, Linear discriminant analysis, or other machine learning models that are descriptive-based, can make the prediction better to be understood, and in cases it works better under certain models that provides mathematical description, like LDA.
3. Since the train & test accuracy rate for the feature importance methods are similar to using ALL of the variables, it can be said that using feature importance is more efficient, as it requires less computational power, because we only solve using 3 variables, instead of all variables, but yielding similar results.



## What we learned?:
1. We learn new methods of machine learning, and perform prediction with high accuracy models, such as neural networks(ANN).
2. We learn to get different features for EDA to get a different outlook for our analysis
3. While we can feature importance from different machine learning, we should not rely on it as the machine learning techniques does not show their inner workings . So we have to use other methods where it requires the use of mathematical formulas to give a more stable conclusion.

## References:
1. Barnett, D. (2022). Jodorowsky animated Dune in development, says crypto group. the Guardian. Retrieved from https://www.theguardian.com/film/2022/jan/24/dune-animation-based-on-jodorowsky-concept-art-in-development-says-cryptocurrency-group-spice-dao-frank-herbert.
2. Raval, D., 2022. Churn Prediction using Neural Networks and ML models. [online] Medium. Available at: <https://towardsdatascience.com/churn-prediction-using-neural-networks-and-ml-models-c817aadb7057>.
3. Variable importance in neural networks | R-bloggers. R-bloggers. (2013). Retrieved from https://www.r-bloggers.com/2013/08/variable-importance-in-neural-networks/.
4. ML | Linear Discriminant Analysis - GeeksforGeeks. GeeksforGeeks. (2021). Retrieved from https://www.geeksforgeeks.org/ml-linear-discriminant-analysis/.

