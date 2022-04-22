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


**2. Data preparation and Exploratory Data Analysis**

As our problem mainly consist of continuous data (dependent variables) as well as categorical data(for independent variable), to see how each of the dependent variable affect the independent variable, we would need to perform visualization on the dependent variables (uni-variate exploration), as well as combinations of some of the variables (bi-variate exploration), with respect to the independent variable of is\_safe. 

To complement the approximation of seeing from approximation of the data, we have also adopted a correlation between each of the dependent variables, as well as with the categorical independent variable, which give us a numerical scale of feature importance as to which variables are likely to predict the is\_safe the best. 

But before that, we need to perform data preparation as well as some exploratory data analysis. Some forms of data cleaning include:

1) txtdata**=**"aluminium - dangerous if greater than 2.8 ammonia - dangerous if greater than 32.5 arsenic - dangerous if greater than 0.01…. 

A string description of the txtdata. The txtdata string shows the indications for the safe and dangerous pertaining to each variable. (**Take not that they are not related to the overall is\_safe water quality, just an indication of the dangerous amounts of each element**)

(But of course, we can use this set of indicators to interpret how do they affect and predict the overall is\_safe, later on)

Where through this, we need to extract  all of the **threshold values** from the abovementioned txtdata file, to be used later to create **binary categorical columns** for each of the variables.


1) As for the data set, we can see that it is highly imbalanced. 

![](Aspose.Words.27acc061-d1be-46c8-b3d1-fed75dd6db93.001.png)

`	`0        7084

`	`1         912

`	`#NUM!       3

`	`Name: is\_safe, dtype: int64



**High imbalance** between 1 and 0, the training model may **overtrain  negatives** and **undertrain from positive ones**. Resulting in **low accuracy** for the positives. Also, we need to remove the #NUM! values as they are irrelevant and obstructive to our computations later on. 


1) Using the info() method to retrieve dataframe information, we noticed that ammonia is inconsistent and has to be changed to float64. If ammonia is an object, then it would not be able to function as a continuous data, that is required for further machine learning and modelling. Therefore, it would be required for us to change the ammonia to float64.

![](Aspose.Words.27acc061-d1be-46c8-b3d1-fed75dd6db93.002.png)



**Exploratory Data Analysis**

As mentioned above, uni-variate exploration, as well as combinations of some of the variables (bi-variate exploration) are required to examine and visualize the data to yield insightful analysis from it.  In this case, we have adopted:

1. Boxplot, histplot, countplot (univariate)
1. Stripplot/swarmplot, density plot(bi-variate with respect to the independent variable)
1. Kdeplot(2 dependent variable compared with each other)

Through the analysis of all the plots, we have come to a conclusion that Uranium, Arsenic and Cadmium have the highest chance of influencing the decision of safe water.

If we look at some of the negative examples: Barium,Bacteria,Nitrates: Such features have **similar percentages** between **Safe and Unsafe**; Inconclusive to predict based on such features.(e.g., bacteria, with an approximate 50% percentage for both safe and unsafe.)

Furthermore, they share similar density plots. For example, for each value (0.0, 0.2, 0.4, etc) of bacteria on density plot, a similar density is observed for safe and unsafe categories, so using difference in terms density between safe and unsafe categories do not lead to meaningful interpretations.

Therefore, such features would not be taken into account.

**Some examples** of **meaningful interpreations** :

- The contrast between density plot, i.e. left-skewed for is\_safe==1 and right skewed for is\_safe==0, then can tell us that the higher the amount of the specific element, the less safe it gets.
- The contrast between boxplots and histplots: is\_safe==0 data for a specific variable has its interquartile range of boxplots smaller than is\_safe==1 (i.e. Q3 for is\_safe==1 for cadmium is even smaller than Q1 for is\_safe==0)
- The mode of the amount of the variable is higher for is\_safe==1, compared to is\_safe==0, at a lower amount. (i.e. for cadmium, although the mode is similar between 0.000- 0.025 for BOTH is\_safe==0 and is\_safe==1, the density for mode(is\_safe==1) is around 20, while density for mode(is\_safe==0) is only at around 7. This shows that for is\_safe==1, there is definitely a higher percetage of the variable with lower amounts, compared to is\_safe==0)

For arsenic, **mode** for safe for Arsenic has **higher density percentage**, at a lower value(left skewed), compared to unsafe for Arsenic. ![](Aspose.Words.27acc061-d1be-46c8-b3d1-fed75dd6db93.003.png)

From the **stripplot**, it can be observed that at higher arsenic values, the density for unsafe is higher as compared to safe, as the scatters for is\_safe==0 is less scarce in comparison with safe.

![](Aspose.Words.27acc061-d1be-46c8-b3d1-fed75dd6db93.004.png)

Therefore it can inferred that higher arsenic values might have correlation with the water being unsafe.

Similarly for cadmium, we can see that the **mode** of safe has a higher density percentages at a lower value, as compared to unsafe. Also the **boxplot** for safe is smaller than for unsafe, as we can see **boxplot** for cadmium (is\_safe==1) has its **Q3**  even smaller than the Q1 for **boxplot** for cadmium(is\_safe==0).![](Aspose.Words.27acc061-d1be-46c8-b3d1-fed75dd6db93.005.png)

For uranium, we can see that the **violin plot** is skewed oppositely with the safe values concentrated on lower ends, and unsafe values on higher ends.

![](Aspose.Words.27acc061-d1be-46c8-b3d1-fed75dd6db93.005.png)


On top of visualisation of the plots, we need concrete and numerical data inference to show the inference and the visualisation of the data is indeed accurate. To show this, we need to perform some forms of calculation which yields in numerical data. In this case, we do it through correlation.

As our independent variable is of **categorical nature**(i.e. **is\_safe==1 || is\_safe==0**), we cannot directly use the **Pearson coefficient** that is between both continuous variables. Instead, point-biserial correlation is used to measure the strength and direction of the association that exists between one continuous variable and one dichotomous variable. It is a special case of Pearson’s product-moment correlation, applied when you have two continuous variables, whereas in this case one of the variables is measured on a dichotomous scale.

For example, a point-biserial correlation can be used to interpret if there is an association between salaries, measured in US dollars, and gender (i.e., your continuous variable would be "salary" and your dichotomous variable would be "gender", which has two categories: "males" and "females").

For our case, comparing all the dependent variables with our independent variable of “is\_safe”, it would be easy to obtain the correlation between the variables and is\_safe.



![](Aspose.Words.27acc061-d1be-46c8-b3d1-fed75dd6db93.006.png)

In this case we pick the **top 3 absolute values** of correlation between is\_safe and the variables. We pick aluminium, cadmium and chromium as feature importance derived from the correlation method.

![](Aspose.Words.27acc061-d1be-46c8-b3d1-fed75dd6db93.007.png) 

![](Aspose.Words.27acc061-d1be-46c8-b3d1-fed75dd6db93.008.png)

Through visualising the 3D plot based on the top 3 feature importance picked from correlation constant, we can tell that  with low cadmium counts, there is a higher density for safe. Also, for lower aluminium count accounts for higher density for unsafe.

3. **Finding Feature importance from Machine Learning Techniques**

On top of “seeing” from the visualisation of EDA, and through correlation, a more accurate prediction method could be determined through machine learning techniques. For that we have picked Random forest and Artificial Neural Network. 

Random forest is an “ensemble” method, incorporating  **many decision trees** and taking an average decision. To obtain a result for an input, the decision process starts from the root node and goes down the depths of the tree until a result leaf is reached. At each node, the path that is chosen depends on the feature value of the specific input, the input value would then follow a path which fits the condition on the tree nodes. 

Repeating a similar process with hundreds or even thousands of decision trees combined and averaged, the Random Forest is less explainable than single decision tree, but typically more accurate.

On the other hand, multi-layered neural networks refers to neural networks with many layers, facilitating “deep learning”. Essentially, these are **compound mathematical functions** that make predictions by minimising error. Neural networks are powerful *due to their* complexity, and have an extremely wide-range of use-cases. They are more precise and accurate as compared to using simpler models. They consist of hundreds to millions of different parameters depending on the size of the network, all interacting in a complex way. This black box problem makes it complicated to use Neural Networks in areas where trustiness and reliability of the predictions are of great importance.

Therefore, if a lot of data is to be processed, but not caring much about explainability, then multi-layered neural networks is a good way to go.

![](Aspose.Words.27acc061-d1be-46c8-b3d1-fed75dd6db93.009.png)

Therefore, we pick random forest, as well as Deep neural networks to predict the feature importance with greater accuracy. The trade-off is that it lowers the explainability of the models, as the description of the processes would not be a matter of the concern, but rather the final prediction outcomes. 

**RANDOM FOREST**

For the random forest, we have decided to use an inbuilt library from sklearn:

**<code>**

**from** sklearn.ensemble **import** RandomForestClassifier

*#Create a Gaussian Classifier*

clf**=**RandomForestClassifier(n\_estimators**=**100)

*#Train the model using the training sets y\_pred=clf.predict(X\_test)*

clf**.**fit(X\_train,y\_train**.**values**.**ravel())

y\_pred\_train**=** clf**.**predict(X\_train)

y\_pred\_test**=**clf**.**predict(X\_test)

feature\_names**=**OriginalDataForLDA**.**columns[0:20]

feature\_imp **=** pd**.**Series(clf**.**feature\_importances\_,index**=**feature\_names)**.**sort\_values(ascending**=False**)

feature\_imp

**</code>**

Through this, we can find the feature importance based on random forest, and we choose the top 3 variables, namely `aluminium`,`cadmium`,`perchlorate`.


![](Aspose.Words.27acc061-d1be-46c8-b3d1-fed75dd6db93.010.png)

**ARTIFICIAL NEURAL NETWORK**

Similar to Random forest, we need to remove missing values and convert all categorical data into numerical ones, we would also need to pre-process the data by scaling all the features into a same range. 

**<code>**

**from** sklearn.preprocessing **import** MinMaxScaler

scaler **=** MinMaxScaler()

OriginalDataForLDA**.**iloc[:,0:20] **=** scaler**.**fit\_transform(OriginalDataForLDA**.**iloc[:,0:20])

X **=** OriginalDataForLDA**.**iloc[:,0:20]

y **=** OriginalDataForLDA['is\_safe']

y**=**y**.**astype(float, errors **=** 'raise')

**</code>**

After processing the data, the architecture of the neural network is created, where the the number of layers (usually 2 or 3) is to be decided, along with how many neurons in each layers and activation functions.  Afterwards, we can also choose to tune the algorithm with an optimiser (optional), where we can tune the learning rate, momentum and decay.

**<code>**

**import** tensorflow **as** tf

**from** tensorflow **import** keras

**from** keras.wrappers.scikit\_learn **import** KerasClassifier, KerasRegressor

**import** eli5

**from** eli5.sklearn **import** PermutationImportance

**def** base\_model():

`    `model **=** keras**.**Sequential([

`    `keras**.**layers**.**Dense(20, input\_shape**=**(20,), activation**=**'relu'),

`    `keras**.**layers**.**Dense(15, activation**=**'relu'),

`    `keras**.**layers**.**Dense(1, activation**=**'sigmoid')

`    `])

`    `**def** get\_optimizer():

`        `**return** tf**.**keras**.**optimizers**.**Adam(learning\_rate**=**0.01)


*# opt = keras.optimizers.Adam(learning\_rate=0.01)*

`    `model**.**compile(optimizer**=**get\_optimizer(),

`              `loss**=**'binary\_crossentropy',

`              `metrics**=**['accuracy'])



`    `model**.**evaluate(X\_test, y\_test)

`    `**return** model


my\_model **=** KerasRegressor(build\_fn**=**base\_model)    

history**=**my\_model**.**fit(X\_train, y\_train, validation\_data**=**(X\_test, y\_test), epochs**=**150)

**</code>**

For this particular model, we have adopted keras.Sequantial() model, which is a linear stack of layers, where new layers are added on top of existing layers. For this model, we have incorporated 3 layers. In the first layer, we have 20 input features(neurons), which gradually passes onto the next layer, and finally to the output layer, with 1 neuron (output, 1 or 0).

In the first layer, we used the ‘relu’ activation function, where the output of individual nodes are are determined as the function output. the ‘relu’ is half rectified function. The output is zero when input is negative and zero and output is equivalent to input when input is bigger than zero. This effectively gets rid of negative data from the features, which are not useful for deriving the overall probability.

For the last layer, we used ‘sigmoid’ activation function to predict the output probability, this is akin to a logistic regression, where the function exists between (0-1), as the input of the neuron gets higher, it will reach closer to 1, and conversely, it reaches closer to 0. Therefore, it can be used to determine the probability of an output, where if the range is closer to 1, it would predict the is\_safe value as closer to 1. 

![](Aspose.Words.27acc061-d1be-46c8-b3d1-fed75dd6db93.011.png)



**To complement the training of data through the layers, we used Adam Optimiser where it** maintains a per-parameter learning rate to improve performance on problems with sparse gradients. Also, it adopts root mean square propagation where it maintains per-parameter learning rates from taking the mean of previous magnitudes of the gradients for the weights. To calculate the loss, we used “binary \_cross entropy” which compares each of the predicted probabilities to actual binary output of 1 or 0. It then calculates the score which penalises the probabilities based on distance from the expected value and therefore making up for losses.

Through the self-training process running through multiple epochs, the accuracy results becomes more optimal with increasing number of epochs, but take note not to overfit, as overfitting can result in decreased test accuracy despite optimal train accuracy rates. As such, using ANN is beneficial to find the most accurate variables for feature importance. 

![](Aspose.Words.27acc061-d1be-46c8-b3d1-fed75dd6db93.012.png)

As seen from the model accuracy, model accuracy is increased with higher number of epochs, but we have to take not to overtrain the data with too many epochs. We can see the prediction accuracy is around 0.96 for the train, and 0.90 at max for the test value. Instead of overtraining the data, we can run at lower epochs of 60-80, before the test accuracy decreases. 

![](Aspose.Words.27acc061-d1be-46c8-b3d1-fed75dd6db93.013.png)

Using the results from neural network, we can then find the feature importance, using a permutation importance, where the **feature importance** is calculated by noticing the **increase or decrease in error** when we **permute the values of a feature.** If permuting the values causes immense changes in the error, it means the feature is important for our model. 

Therefore, the top 3 features chosen from feature importance by using ANN is `aluminium`,`perchlorate`,`silver`.


3. **Using top 3 variables of feature importance derived from each method, perform prediction using Random Forest and Linear Discriminant Analysis, and compare with prediction using all 20 variables.**

Top three variables from feature importance using EDA: `uranium`, `arsenic`,`cadmium`

Top three variables from feature importance using Correlation: `Aluminium`, `cadmium`, `chromium`

Top three variables from feature importance using Random Forest: `Aluminium`, `Cadmium`, `perchlorate`

Top three variables from feature importance using Artificial neural network: `Aluminium`, `perchlorate`, `silver`

From each of the sets of top three variables, we hereby perform machine learning models to the top variables only, via random forest and Linear discriminant analysis. From here, we can deduce which of the sets of top three variables can yield the best prediction accuracy. Also, How does the prediction accuracy of the selected variables based on feature importance(using top 3 variables only) compare with prediction using **all** variables? 

As abovementioned, we use random forest because it yields highly accurate results; the complexity of incorporating and averaging many decision trees into the model undoubtedly makes the model more accurate. However, such model poses some challenges too; it only predicts the model to a high degree of accuracy, but having such a model does not allow us to understand how to derive the steps of solving the problem; in other words, such random forest lacks explainability, which is vital for certain forms of data interpretation, for example:

**1. Accountability:** When a model predicts wrong, we need to account for the factors inducing the decision and responsible for failure, such that it enables us to mitigate similar problems in future. 

**2. Trust**: In high confidentiality domains like finance, trust is critical. Before ML solutions can be trusted, all stakeholders must fully understand the exact functioning of the model. Therefore, it is needed to be backed up with evidence to prove the functioning of the models before that they can be used. 

**3. Performance**: We can fine-tune and optimize our models to improve the performance if we understand the functioning of the models. 

As such, we decide to adopt another machine learning model with enhanced explainability, but not necessarily better prediction results as compared to random forest. Afterall, our aim is not to achieve the best prediction result from using a particular model. Rather, we strive to see if using 3 variables from feature importance is comparable to using ALL 20 variables. Adding another machine learning model is just for interpreting the problem in a more explainable nature, and to see if the top 3 variables using feature importance can ensure comparable prediction results as with all 20 variables for another model. 






**Linear Discriminant Analysis**

We have chosen Linear Discriminant Analysis as the second model, with enhanced explanability due to its strong mathematical basis. It is a dimensionality reduction technique which projects the features in higher dimension space to lower dimensions. 

![](Aspose.Words.27acc061-d1be-46c8-b3d1-fed75dd6db93.014.png)![](Aspose.Words.27acc061-d1be-46c8-b3d1-fed75dd6db93.015.png)![](Aspose.Words.27acc061-d1be-46c8-b3d1-fed75dd6db93.016.png)

The advantage of LDA is that it creates a new line of a lower dimension from the previously 2 dimension data as shown above. Then LDA uses information from both features to create a new axis which can in turn minimise the variance and maximise the class distance between the two features.  

Having such a model is explainable because it uses a set of explicit of mathematical procedures, so that we can explain how to derive the model. Also, as it is a dimensionality reduction technique, it enables a reduction in dimensional cost and resources, which also adheres to our practical motivation and problem statement. 

By reducing the dimension of a set of 3 features into a 2D plane, we can see that reduction of the 3 features into a 2D plane enables the features to be more well split, with the is\_safe==1 (red)values more to the left and the is\_safe==0(blue) values more towards the right.![](Aspose.Words.27acc061-d1be-46c8-b3d1-fed75dd6db93.017.png)

Thus, for each set of the top 3 variables from feature importance (using EDA, Corr, Random Forest and ANN), we predict the is\_safe value using Random Forest and Linear Discriminant Analysis. We have tabulated a table of results to show the prediction accuracy under all different circumstances. 



























|<p> </p><p> </p>|**With All 20 variables**|**(Top 3 features) With feature Importance**|
| :- | :- | :- |
|||<p>**(Top 3 features**</p><p>`  `**derived using each of the methods)**</p>|
|** ||**EDA**|**Correlation**|**Random Forest**|<p>**Neural**</p><p>`  `**Network**</p>|
|**Model of Prediction** |RF|LDA|RF|LDA|RF|LDA|RF|LDA|RF|LDA|
|**Classification Accuracy**|1|0.783|0.92|0.743|0.993|0.773|1|0.795|1|0.772|
|**True Positive**|676|523|591|` `495|684|` `560|676|` `602|669|551|
|**True Negative**|692|548|`   `662|` `522|675|` `498|692|486|699|` `505|
|**False Positive**|0|` `155|84|` `189|0|` `117|0|` `86|0|` `130|


It can be seen that, through using all 20 variables, and as compared to only using 3 features from the respective feature importance methods, we can see that the classification accuracy are actually comparable, where top 3 features from neural network, random forest and correlation yields 1, 1 and 0.993 respectively, compared to 1 for all 20 variables, this shows that using 3 features is as effective in terms of prediction accuracy as compared to ALL 20 variables, while saving resources and computational cost. 

Also, we can see that although top 3 feature importance from neural network yield a good result through Random forest prediction, classification accuracy for LDA (0.772) is not as optimal as compared to compared from top 3 feature importance using correlation (0.773) and random forest (0.795). This shows that while obtaining feature importance from a particular method might be good fit for one model, but not the other. This shows that, we should not limit ourselves to using only one method of obtaining feature importance, but rather we should try different alternative, and use different sets of features for different model predictions.



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
