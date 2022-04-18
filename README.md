# DataScience-Proj

### Done by: 
1. Muhammad Nabil Hakeem
2. Su Gaoyang
3. Scott Wong

## Background:
Access to safe drinking water is important so that there is no health issues that arises from drinking contaminated water. Globally, 772 million people still lack basic access to clean water.There continues to be an investment in developing proper drinking water at a global level. However, there can be many factors that affect the condition of the water. Water samples are regularly collected from reservoirs and waterworks to ensure water supply is clean and safe for everyone. With so many water sources around the world, a lot of resources are spent on testing.

## Please view the codes in this order:
1. Data Extraction And Data Preparation (EDA: https://github.com/Note06/DataScience-Proj/blob/main/1015-Proj%20EDA.ipynb ) / (Correlation: https://github.com/Note06/DataScience-Proj/blob/main/1015-Proj%20EDA%2BCorrelation%20(2).ipynb )
2. Using Machine Learning to get feature importance (Random forest: https://github.com/Note06/DataScience-Proj/blob/main/1015-Proj%20EDA%2BRandom%20Forest%20(3).ipynb ) /  (Neural Network: https://github.com/Note06/DataScience-Proj/blob/main/1015-Proj%20EDA%20%2B%20Neural%20Network.ipynb )
3. We then use the data identified from EDA, Correlation, Random Forest and Neural Network ,and apply it to different prediction models. (EDA: https://github.com/Note06/DataScience-Proj/blob/main/Prediction%20with%20Feature%20Importance(EDA).ipynb ) / (Correlation: https://github.com/Note06/DataScience-Proj/blob/main/Prediction%20with%20Feature%20Importance(Correlation).ipynb ) / (Random Forest: https://github.com/Note06/DataScience-Proj/blob/main/Prediction%20with%20Feature%20Importance(Random%20Forrest).ipynb ) / (Neural Network: https://github.com/Note06/DataScience-Proj/blob/main/Prediction%20with%20Feature%20Importance(NeuralNetwork).ipynb )


## Problem Statement: 
With so many different factors that can affect water quality, which ones have the most impact?<br>
We decide to use the idea of feature importance. By identifying data that impacts the decision the most we can increase our data accuracy prediction and improve efficiency. 

## Dataset from:
MsSmartyPants. (2021, June 30). Water quality. Kaggle. Retrieved February 3, 2022, from https://www.kaggle.com/datasets/mssmartypants/water-quality

## Models Used:
1. Random Forest
2. Neural Network

## Conclusion:
1. The 2 variables that appear the most number of times are Aluminium and Cadmium. From this, we can tell that the 2 variables are key to ensuring that the water is safe to drink.
2. Using ANN to get the feature importance yields good results under random forest prediction, but not as optimal for LDA. And furthermore, ANN doesn't tell you the process of prediction. So for certain predictions could not be explained and described scientifically. Therefore, resorting to other models, like through examining Correlation, can make the prediction better to be understood, and in cases it works better under certain models that provides mathematical description, like LDA.
3. Since the train & test accuracy rate for the feature importance methods are similar to using ALL of the variables, it can be said that using feature importance is more efficient, as it requires less computational power, because we only solve using 3 variables, instead of all variables, but yielding similar results.



## What we learned?:


## References:

