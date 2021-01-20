![alt text](https://github.com/iceberg425/Reddit_API_Project/blob/master/Reddit_Post_NLP/images/TVshowsMovieswordle2.png "Logo Title Text 1")


### Workflow:

1. [Data Gathering](https://github.com/iceberg425/Reddit_API_Project/blob/master/Reddit_Post_NLP/data_gathering.ipynb)
2. [Data Exploration](https://github.com/iceberg425/Reddit_API_Project/blob/master/Reddit_Post_NLP/EDA.ipynb)
3. [Pre-processing and Modeling](https://github.com/iceberg425/Reddit_API_Project/blob/master/Reddit_Post_NLP/Modeling.ipynb)
4. [Model Evaluation](https://github.com/iceberg425/Reddit_API_Project/blob/master/Reddit_Post_NLP/Further_Evaluation.ipynb)
5. Concluding Remarks


### Problem Statment

   
  For anyone familiar with television shows or movies, it is easy to identify if the Office is a TV show or not. Unlike humans, a computer cannot easily distinguish words that are associated with either a movie or TV show, or how to categorize them. However, with the appropriate training, a computer should be able to identify such words as a human does.

 This scenario is similar to the problem at hand. Let's assume that a start-up is attempting to launch a new product that can identify which words are associated with a TV show or a movie. That is, the problem at hand is a classification problem: is a word (or set of words) related to a tv show or not.

 Now assume that the project manager communicates this to you, as the only data scientist on the team (assuming it's a very small start-up), and assigns you the responsibility of training a model for this task. However, she does not provide you with any data, a key component of model building. After an extensive google search, you do not find any appropriate structured data for the task. You, therefore, resort to gathering unstructured data from Reddit. The first task is to gather the data, engineer it so that it can be transformed into structured data, and then develop, train the model, and evaluate the model. 


### Executive Summary

 In this project, we are faced with a classification task. The goal is to develop a model that can accurately place words in the right category. That is, we want to train a model that sees a word and correctly tells us if it is related to tv or not. 
   
   Since we do not have structured data at hand, we are left to gather unstructed text data. This kind of data never comes in 'clean'. It involves several iteartions of cleaning and parsing, which was the case in this project. 
   
   The data employed here was gathered from Reddit. We tackle the problem using movie and television subreddits. Given the restriction on the amount of data that can be retrieved per time, we collected 1231 movie posts and 2174 tv posts, making a total of 3405 posts. In more technical terms, we have 3045 documents of text data to analyze.
   
   We employ three algorithms and several models (due to hyperparameter tuning). 
   - Logistic Regression 
   - Support Vector Classifier
   - Gradient Boosting Classifier
   
   The Logistic regression returned an 83% accuracy, the Support Vector Classifier returned an accuracy of 84%, and the Gradient Boosting classifier an accuracy of 88%. 
   
   We eventually settle with one model: the Gradient Boosting model. This model tends to generalize fairly well. When we take this model to unseen data, we find that it performs just as well as it did when it was trained.  
   
   Using this model leaves us with the challenge of interpretability. That is, these kind of models, unlike logistic regressions and linear regressions, do not provide us with coefficients that can be easily interpreted. If our interest is to draw inference, then a logistic regression would be our best bet here. However, since the goal is not inference but accuracy, the lack of interpretation is not much of a problem. 
   
   The GB model generalizes well and outperforms the Logistic Regression. Therefore, we settle on this model and recommend it for deployment for the task.  


### Data Gathering

   The data employed here was gathered from Reddit. We tackle the problem using movie and television subreddits. Given the restriction on the amount of data that can be retrieved per time, we collected 1231 movie posts and 2174 tv posts, making a total of 3405 posts. In more technical terms, we have 3045 documents of text data to analyze.

   The code for the data gathering process is not reported here. Please see the Jupyter Notebook titled data_gathering for the details of the code.

   After gathering the data, we randomly split the data into a traina and test split (or a holdout set). This was done to test our model's generalizability. When faced with new data, would our model perform just as well as it did when we trained it? We attempt to address this question in a convincing way.
    
   

### Data Exploration

   We identify missing data. For a number of cells, there is no text data even though we identify that users made comments related to a post. This activity can be seen in the `num_comments` column. This suggests that users posting comments may not have used words. Instead, they most likely posted a pic, gif, or even some other form of media. This is common in subreddits in general.
   We address this by conducting some feature engineering. By merging the title and the associated comments, we now have documents with text that can be used in our analysis. Furthermore, we binarize the `subreddit` column.Please see the data dictionary for a description of this binary variable.  

### Pre-Processing and Modeling

   To pre-process the data, we begin by placing our text data into a dataframe. This is done for two reasons: 1) to view our data more easily 2) to easily manipulate the data and prepare the data for modeling. The preprocessing is in five stages: 
   1. We remove html code from the text
   2. We use Regular Expressions (regex) to remove punctuations and other none-words
   3. We remove "stop-words". That is extremely common words which would be of little value in the analysis.
   4. The final step, which is transforming the data into values that a computer csan understand is done alongside our modeling, since we use a Pipeline and grid-search.
   
   We employ four algorithms and several models (due to hyperparameter tuning). We eventually settle for two models: the Support Vector classifier and a Gradient Boosting model. These models tend to generalize fairly well. 
   

### Model Evaluation
   
   We evaluate our model's performance by analyzing its accuracy, misclassification rate, recall and precision. After an initial cross-validation, scoring and predicting on the test set, we took the model to unseen data (the validation set). The GB model performed just as well as it did on the training and test data. Unlike the Logistic regression, there is no clear evidence of overfitting. 
   

### Concluding Remarks

   Using these models leaves us with the challenge of interpretability. That is, these kind of models, unlike logistic regressions and linear regressions, do not provide us with coefficients that can be easily interpreted. If our interest is to draw inference, then a logistic regression would be our best bet here. However, since the goal is not inference but accuracy, the lack of interpretation is not much of a problem.      

   
   






```python

```
# classifier_poc
