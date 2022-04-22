# SC1015 DataScience-Proj

### Done by: 
1. Muhammad Nabil Hakeem (U2122831K)
2. Su Gaoyang (U2121723E)
3. Scott Wong  (U2122366G)

## Background:
Let us first take a look into the global Drinking water quality issue.
•	Globally, 772 million people lack basic access to clean water.
•	The number of deaths due to unsafe water is 485,000 in 2020.
•	With so many different water samples around the world, a lot of resources and computational powers are spent to test on factors affecting the quality of water.

With this regard, is it necessary to test the quality of water based on ALL of the available variables in the water sample? Including ALL of the available variables may become a time-consuming process and demand a lot of computational power, which is certainly not viable for the prevalent testing of the water. In an attempt to improve efficiency and to improve prediction accuracy of testing of the water samples, we decide to look into feature importance,  where feature Importance refers to techniques that calculate a score for all the input features for a given model — the scores simply represent the “importance” of each feature. A higher score means that the specific feature will have a larger effect on the model that is being used to predict a certain variable. From the set of the scores, we can then sieve out the most important features, for prediction. 

## Why Feature Importance is useful
1. Data Understanding - Like a correlation matrix, feature importance allows us to understand the relationship between the various dependent variables and the independent variables. It enables us to understand what features are irrelevant, so that we can sieve them out.
2. Model Improvement - The scores calculated from feature importance can help to reduce dimensionality of the model (LDA and PCA). The higher scores are kept and lower scores that are deemed as less correlated with the model are removed. This simplifies the model, and also makes the model more efficient, improving the performance of the model. As such, selecting the appropriate features for these models is important. Irrelevant data culminates in bias, potentially decreasing accuracy of our prediction. Excessive columns of data in the dataset takes too much computational power and time, inefficient and redundant.

## Problem Statement: 
With so many different factors that can affect water quality, which ones have the most impact?<br>
We decide to use the idea of feature importance. By identifying data that impacts the decision the most we can increase our data accuracy prediction and improve efficiency. 

## Dataset from:
MsSmartyPants. (2021, June 30). Water quality. Kaggle. Retrieved February 3, 2022, from https://www.kaggle.com/datasets/mssmartypants/water-quality

## Please view the codes in this order:
1. [Data Extraction And Data Preparation](https://github.com/Note06/DataScience-Proj/blob/main/1015-Proj%20EDA.ipynb)
2. [Using Machine Learning to get feature importance](https://github.com/Note06/DataScience-Proj/blob/main/All%20variable%20machine%20learning.ipynb)
3. [Prediction Models](https://github.com/Note06/DataScience-Proj/blob/main/Prediction%20using%20Feature%20Importance.ipynb)


## Models Used:
1. Random Forest (In presentation, used for finding feature importance, and Prediction of the model)
2. Neural Network (In presentation, used for finding feature importance)
3. K Nearest Neighbours (Never discuss in the slide, but performed the model)
4. Logistic Regression (Never discuss in the slide, but performed the model)
5. Linear Discriminant Analysis & Principle Discriminant Analysis (Used for prediction of the model 

## Conclusion:
Top three variables from feature importance using EDA: 'uranium', 'arsenic','cadmium'
Top three variables from feature importance using Correlation: 'Aluminium', 'cadmium', 'chromium'
Top three variables from feature importance using Random Forest: 'Aluminium', 'Cadmium', 'perchlorate'
Top three variables from feature importance using Artificial neural network: 'Aluminium', 'perchlorate', 'silver'

1. The 2 variables that appear the most number of times are Aluminium and Cadmium for top 3 feature importance using the 4 different methods. From this, we can tell that the 2 variables are key to ensuring that the water is safe to drink. It is vital to maintain and optimal amount of aluminium in the water, and decrease the amount of cadmium in water to as low as possible. High feature importance across various models suggest that such variables have a higher tendency of affecting the prediction of the independent variable, hence they should be “tackled”. Such observations could be well applied for other datasets, as the idea of feature importance is “universal” in the domain of data science. 

2. Using ANN to get the feature importance yields good results under random forest prediction, but not as optimal for LDA. Furthermore, through ANN's self-learning, it doesn't tell you the process of reasoning on derivation of the results. Therefore, we should obtain feature importance using different machine learning models. In our data, obtaining feature importance through random forest and correlation yields higher classification accuracy than using LDA. Furthermore, blindly trusting the numerical information is undesirable too, as some of the numbers do not provide a clear visual representation of the data for us to interpret and sieve out irrelevant data, therefore, we have to validate and support the results via EDA and correlation that are descriptive-based, can make the prediction better understood.

3. Using 3 Variables through feature importance across all of the 4 methods yields comparable results (classification accuracy) compared with using ALL 20 variables. Therefore, it can be said that using 3 carefully selected variables is sufficient for data accuracy and more efficient in computational power. With this insight, we can probably adopt such a method on datasets with huge amount of data (i.e. 1000 columns), as running through all columns of the datasets would be unrealistic as it incurs high time complexity and computational power.


## What we learned?:
1. There is no one-size-fits-all approach in terms of picking feature importance with respect to a particular method to determine feature importance. We should adopt different methods, through both machine learning and visual forms of exploration through exploratory data analysis, and predict with different sets of feature importance using machine learning methods. Furthermore, using EDA’s variety of visualisation tools, can enable data scientists’ to better validate and analyse the top features from feature importance using the different methods. From here, we have learnt new methods of machine learning, and perform prediction with high accuracy models, such as neural networks. We have also learnt to extract different features from EDA to obtain a different outlook for our analysis. 
2. Although some models(like neural networks, random forest) are able to provide highly accurate predictions, it is only predictive in nature, with the lack of description for obtaining the prediction, thus attributing to the lack of explanability of the models. Therefore, if we only want to obtain a prediction based on the available data, then having such models are fine. However, if we strive to seek an explanation of the model, with an analysis of its innerworkings and mathematical procedures( for example, to detect the procedure which constitute to the wrongful prediction, or to track the parts that can be fine-tuned or improved), then we might need to resort to other models which feature greater explanability.
3. Feature importance is a powerful tool to extract important features and remove redundant and irrelevant features. It would be immensely useful if we are dealing with datasets containing a lot of features, as removing certain parts of them clears the “noise” or distraction, while enabling the model to focus more on the relevant datasets. And for our dataset, it can be seen that despite only using 3 variables, our models of predictions actually have comparable classification predictions as compared to using all 20 variables, which shows that it is indeed viable to extract prominent features from a dataset to undergo prediction. 
4. Enhanced ways of data preparation for adoption on different models, for example, the Artificial neural network requires `MinMaxScaler`, where estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between (0-1).  Also, the ` RandomUnderSampler` enables us to Under-sample the majority class(es) by randomly picking samples without replacement. Such methods are important as they facilitate for case-by-case data preparation adaptable for different models, which is important in the case of adopting more than one models for prediction. 

## References:
1. Barnett, D. (2022). Jodorowsky animated Dune in development, says crypto group. the Guardian. Retrieved from https://www.theguardian.com/film/2022/jan/24/dune-animation-based-on-jodorowsky-concept-art-in-development-says-cryptocurrency-group-spice-dao-frank-herbert.
2. Raval, D., 2022. Churn Prediction using Neural Networks and ML models. [online] Medium. Available at: <https://towardsdatascience.com/churn-prediction-using-neural-networks-and-ml-models-c817aadb7057>.
3. Variable importance in neural networks | R-bloggers. R-bloggers. (2013). Retrieved from https://www.r-bloggers.com/2013/08/variable-importance-in-neural-networks/.
4. ML | Linear Discriminant Analysis - GeeksforGeeks. GeeksforGeeks. (2021). Retrieved from https://www.geeksforgeeks.org/ml-linear-discriminant-analysis/.
5. Bushaev, V. (2018, October 24). Adam - latest trends in deep learning optimization. Medium. Retrieved from https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c
6. How to choose the right AI model for Your application: Obviously AI blog. Data Science without Code. (n.d.). Retrieved from https://www.obviously.ai/post/how-to-choose-the-right-ai-model-for-your-application
7. Onose, A. E., Ejiro Onose Machine Learning Engineer and Researcher, Onose, E., Researcher, M. L. E. and, & on, F. me. (2021, December 13). Explainability and auditability in ML: Definitions, techniques, and Tools. neptune.ai. Retrieved from https://neptune.ai/blog/explainability-auditability-ml-definitions-techniques-tools#:~:text=Explainability%20in%20machine%20learning%20means,applies%20to%20all%20artificial%20intelligence.
8. Point-biserial correlation using SPSS statistics. Point-Biserial Correlation in SPSS Statistics - Procedure, assumptions, and output using a relevant example. (n.d.). Retrieved from https://statistics.laerd.com/spss-tutorials/point-biserial-correlation-using-spss-statistics.php 
9. Sharma, S. (2021, July 4). Activation functions in neural networks. Medium. Retrieved from https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
10. Shin, T. (2021, February 26). Understanding feature importance and how to implement it in Python. Medium. Retrieved from https://towardsdatascience.com/understanding-feature-importance-and-how-to-implement-it-in-python-ff0287b20285#:~:text=1)%20Data%20Understanding.&text=Like%20a%20correlation%20matrix%2C%20feature,are%20irrelevant%20for%20the%20model. 
11. What is linear discriminant analysis(lda)? What is Linear Discriminant Analysis(LDA)? (n.d.). Retrieved from https://www.knowledgehut.com/blog/data-science/linear-discriminant-analysis-for-machine-learning 

