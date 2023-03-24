<!-- Eduardo Spiegel and Trey Scheid -->
<img src="assets/healthy_food.png" alt="eatyoveggies" height=546 style="display: block; margin: 0 auto">
Photo by <a href="https://unsplash.com/@hermez777?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Hermes Rivera</a> on <a href="https://unsplash.com/photos/Ww8eQWjMJWk?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
<!-- # The Recipe for Success: Predicting Balanced Meals
For UCSD class DSC80 project 5 -->
<br>

## Introduction

> Our exploratory data analysis (and a fun hypothesis test) on this dataset can be found <a href="https://trey-scheid.github.io/Recipe-Healthyness-Trends-by-season/" target="_blank">here</a>.

In this notebook we are going to be analyzing a dataset that contains recipes and ratings from <a href="https://www.food.com/" target="_blank">Food.com</a>. In <a href="https://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19c.pdf" target="_blank">this paper</a>, Bodhisattwa Prasad Majumder\*, Shuyang Li\*, Jianmo Ni, and Julian McAuley scraped Food.com for approximately 180K recipes and 700K interactions[^1]. Our University professors have given us a subset; the combined dataset before cleaning consists of `234,429` interactions on known recipes.

[^1]: <a href="https://aclanthology.org/D19-1613" target="_blank">Generating Personalized Recipes from Historical User Preferences</a> (Majumder et al., EMNLP-IJCNLP 2019)

In <a href="https://trey-scheid.github.io/Recipe-Healthyness-Trends-by-season/" target="_blank">our previous project</a> which we are building upon, we were interested in answering this question: Do people eat more unbalanced food during the winter holiday season? The results were not significant in favor of our hypothesis that people do eat more unbalanced food then. Next we want to try and predict:

**Can we predict if a recipe is balanced without knowing its nutrition information?**

For this we are going to analyze the columns `'ingredients'`, `'tags'` and many more to predict `'balanced'`.

Other columns we might work with during and after the cleaning process are `'minutes'`,`'submitted'`,`'n_steps'`, and `'n_ingredients'`. They contain relevant information about the recipe that when extreme, raises some red flags to be explored.

<br>

# Framing The Problem

This is a binary classification problem since our response variable has only two outputs True or False; a recipe is either balanced or not balanced. The EDA defines balanced recipes as ones that have nutrition info from 5-20% Daily Value. We will explore that further shortly.

__Importance:__ There is practical use in knowing if a recipe is balanced or not. Things like ingredients and steps are common information about a recipe that even your grandmas best will have. Nutritional specifics are not as easy to get and thus may not always be available, in such a case if we can predict if it would still be a balanced recipe then you can make healthy recommendations. This might be done in an app or website with a search feature and a "healthy" tag or something similar. 

__Metric:__ We think it is more important that we don't incorrectly classify a recipe as balanced when it is not, this might lead to recommendations to people for healthy recipes that are not actually healthy! These cases we want to avoid are false positives so we want to maximize precision. Some other metrics such as accuracy are missing a level of detail while the rest are including things that aren't critical such as false negatives in confusion matrices and F1 scores. 

We will not be using any interactions (`'review'` or `'avg_rating'`) on the recipes since these are posted onto the recipe after they are posted to Food.com, we would like to classify before users try the recipes!

<br>

### Balanced Recipes

In this project we are primarily concerned with the healthiness of a recipe, more scientifically we describe it as "balanced". The US government's Food and Drug Administration has guidelines on what makes a food healthy. We will review this then create our own specific definition.

>Under the new guidelines, something that is “healthy” needs to have the equivalent of a serving of fruits, vegetables, grains, dairy or protein foods as indicated in the Dietary Guidelines for Americans. Raw, whole fruits and vegetables automatically can bear the claim. There is a scale for different kinds of prepared products that has a nutrient requirement and percentage limits for the recommended daily intakes of added sugars, sodium and saturated fats.  
This definition, which has been six years in the making, is a comprehensive and multifaceted way to handle the label claim, which many consumers would readily accept as an indicator that a food item is unequivocally good for them.

https://www.fooddive.com/news/fda-healthy-definition/632661/

>FDA regulates about 78 percent of the U.S. food supply. This includes everything we eat except for meat, poultry, and some egg products.

https://www.fda.gov/about-fda/fda-basics/fact-sheet-fda-glance

(USDA regulates the rest, they have their own <a href="https://www.fooddive.com/news/usda-sets-parameters-for-items-labeled-healthy/574387/" tag="blank_">rules</a> for healthy items)

According to the FDA, meals should contain between `5%` to `20%` of the percentage of daily value (PDV). 

https://www.fda.gov/food/new-nutrition-facts-label/lows-and-highs-percent-daily-value-new-nutrition-facts-label

We will alter it slightly to better represent what is generally percieved as healthy and the more recent definition by the FDA. That means less than 5% sugar or fat is good, and greater than 20% protein is also healthy. We added a column to the data which is True or False based on the information from the nutrition column. 

<br>

### Additional Cleaning

While cleaning in the last project we checked out the distribution in many columns performing datatype conversions, outlier removal and think about the true data generating process while we do it.

We looked closer into the ingredients column since it will be important for our prediction.

We have seen that many ingredients are written in their plural and singular forms. This underrepresents their prescence in recipes since their count is distributed as two different ingredients. Therefore, to fix this we decided to strip all letters "s" from the end of the ingredients. For example this combined 'egg' and 'eggs' into one column 'egg'. 
<!-- Here are the first few rows of the cleaned dataframe:

<iframe src="assets/sdatahead.html" width=900 height=210 frameBorder=0 title="cleaned dataset preview"></iframe> -->

<br>
<br>

# Machine Learning: Classification

__The Process__: <br>
We will choose some _features_ that we think apply best to predicting `balance` of a recipe, then _train_ a few models using the default hyperparameters. Once we see their precisions (calculated using a test dataset we set aside) we will _tune_ the best one or two models, depending on how similar their scores were, and try to train the best model possible. Tuning will involve _cross validation_ with a randomized/grid searches of possible combinations of hyperparameters. Finally we will have a model fitted on the full dataset you can try on your own! 

Keep in mind, if a _simple_ solution works as well as a complicated one the simple solution is preferred. Thats why we will start with few features and slowly add more checking for improved performance.

<br>
<br>

### Feature Selection

We must remove columns that are ineligible according to the data generating process (GDP).

Our model should be used to predict the balanced label as `True` or `False` given information about the recipe except the nutrition. We are assuming we won't have any `'review'`s on it (under different assumpotions you could create a model for that as well). This allows us to prepare for the person uploading can be someone new to Food.com or not. If our model trained on previous `'contributor_id'`'s this may help if existing users post new recipes but not for general cases and new users, therefore we will overfit! The `'nutrition'` info itself is equivalent to creating balanced so that must also be removed.  

`'Name'`, `'submitted'`, `'steps'`, and `'description'` all cannot be directly used in the model because they will lead to overfitting, they are generally too unique to each recipe but we may be able to create features using them!

We will create these features using the sci-kit learn package transformers and drop the remaining columns in the pipeline!

Checking for missing data we found only one column we are using had missing values: `'description'`. We will remove the recipes missing a description because there are so few.

<br>

# Baseline Model

For the base model we will start simple.

Standardize (quantitative):
- minutes
- n_steps
- n_ingredients

One hot encode the unique values in the ingredients list (categorical)

<br>
### Data

A good practice for a classification problem is to inspect your training data by class. We checked and found that only 10% of the recipes in our dataset are balanced!

From an unknown online source, "Currently, about 5 percent of all packaged foods are labeled “healthy,” according to the agency." So we shouldn't worry that our recipes dataset was biased in this regard.

Even though it reflects reality this is a severe imbalance in classes that still poses a problem for our models. They have less data to train on in the most important category. Our performance measure precision should not be as affected by this like accuracy would, however we still need to address it. There are three ways to deal with the imbalance. 
1. Get more data (ideal, but not an option)
2. Use a model that naturally deals with the imbalance
3. Resample our training data to have equal sized classes through under or oversampling methods. 

We will try 2 and 3, but we need to do a few things first.

First thing we must do, set aside a testing dataset. This is paramount, we can then test our models as we go on unseen data which will give us a better picture on how well it generalizes! We are splitting 15% for testing and made sure the split was stratified to ensure both sets have appropriate class balances that reflect the whole sample we were given. 

`size of training set:  59967`<br>
`size of test set:  10583`

<br>

### Preprocessing

Second, build a preprocessing pipeline. This is a lot to look at so we will break it down, which is the whole idea of a pipeline to streamline the process but keep it manageable. Looking top-down at our pipeline there is a split for the categorical vs quantitative variables (the ones we want are used and the other columns dropped). Then the next level we standardize all the quantitative vars and one hot encode the values in the ingredients list with separate transformers. Standardizing is needed because we don't want minutes having a greater impact on the model simply due to units. Encoding the ingredients lists is an abnormal format which sklearn has a transformer for, but not meant for pipelines from our research, we altered it by creating our own custom transformer class using their encoder the MultiLabelBinarizer under the hood. You will notice we also have two additional transformers on ingredients, these reduce the number of columns/features we keep from the encoder so the model isn't trained on too many columns. The VarianceThreshold will keep only the features with at least some threshold of variance, for a one hot encoded column that means that its neither overly popular or overly rare but somewhere in the middle. The next transformer selects the top percentile of columns by their ability to score well on prediction (usefullness!).

The MultiLabelBinarizer creates a column for each unique ingredient and one hot encodes it for every recipe. Sci-Kit learn does not have pipeline support from our understanding. What we do is save the unique ingredients accross all recipes when fitted, then on all future transforms the output features dataframe will only include those columns (dropping any new ingredients, and adding 0 cols for the unseen ingredients from fit). Our class creates many warnings about fragmentation, thats why we suppressed them, it is not a perfect implementation but we avoided for loops :).

<br>

### Training

Remember we are going to try two different methods for dealing with the class imbalance. For each we will add a model to the pipeline (by creating a new one and using the preprocessing pipeline within). Then We will fit it with the training data and see how it performs on the testing data. We are doing this with mostly default arguments and then we will pick one that works well to fine tune.

<br>

_Method 1:_ test models that handle unbalanced data

**Precision Results**<br>
LogisticRegression     0.13169341532923354<br>
DecisionTreeClassifier 0.13376383763837638<br>
RandomForestClassifier 0.1486030089038993<br>
SVC                    0.14330474934036938<br>
XGBClassifier          0.0<br>
SGDClassifier          0.13209915280828366<br>

Overall the precision is dissapointingly low. Precision is a proportion so it ranges from [0, 1] where 1 would be only correct positive predictions and 0 is none. Almost all the models performed at the same level which makes it difficult to reason why to tune one over another, maybe we will see different results with resampling.

<br>

_Method 2:_ test models on undersampled data

Both under and over sampling have the goal of training a model on balanced classes so that it can notice and recognize both, but we will still test the precision / performance of each model on the imbalanced true testing set to get the best idea of how it will generalize to unseen data.

Undersampling is when we randomly choose n rows to keep from the larger class (not balanced in this case) so that both our classes for training are of equal size.

Oversampling is similar except to achieve balance we bootstrap (sample with replacement) from the smaller class, balanced, until they are equal sizes.

**Undersampled Precision**<br>
KNeighborsClassifier      0.12097476066144473<br>
GaussianProcessClassifier 0.14451656986675776<br>
<br>
**Oversampled Precision**<br>
KNeighborsClassifier      0.12097476066144473<br>
GaussianProcessClassifier 0.14451656986675776<br>

Due to "high" precision in comparison to the others, and very fast training time which we observed, we chose the RandomForestClassifier model to continue with and tune. We also did a few runs and some others that performed as well were not as consistent.

<br>

### Fine Tuning

There are many hyperparameters to tune between the preprocessing pipeline and the Random Forest. Due to time constraints we will perform the searches separately but ideally you would run a grid search checking all the combinations. We also will only use 3 cross-validation folds to limit the amount of training being done.

The searcher takes in a dict of hyperparameters which describe paths down to the correct pipeline item and param then a list of options to try. We will define these then run it. We are using a new Halving grid search that uses subsets of data to narrow its search at the start, with a large dataset and limited compute resources we have to make sacrifices! That and only using 3 cross validation folds, mean that we are less confident in the final parameters that the searcher finds. We also are not going to be giving it many to search through so those other issues may not be as significant anyway. 

_tune preprocessor_<br>
{'preprocessor__ingredients__select_percentile__percentile': 100,<br>
 'preprocessor__ingredients__variance_selector__threshold': 0.004975000000000005}<br>
 
 The first thing to notice, percentile is 100, so it is saying to keep all the columns. Going forward we will remove this transformer entirely. If we had more time we would test the performance with a seacher that performs SelectPercentile before VarianceThreshold, or simply without it, to see which is better.

**Tuned Precision**<br>
preproc_tuned 0.4358974358974359<br>
model_tuned   0.48863636363636365<br>

Interestingly the preprocessing had a much larger difference, either way in total we see big improvements to precision. This was the goal! We only tuned 3 hyperparameters total, with more resources you could test many more combinations.

Double check results<br>
True Precision: 0.45121951219512196<br>
True Recall:    0.03259911894273128<br>
Accuracy:       0.8919965983180572<br>

<br>

When using Random Forests or Decision Trees people often talk about the parameter: max_depth. We checked the forest and found that using auto, the tallest tree has a max_depth of 223. Then we check and found that the forest saw 290 features from the preprocessor when being fit. This isn't a very detailed look but the max_depth does not seem outrageous if it asks on average less than one question per feature. We can safely move on without further tuning the max_depth hyperparameter.

<br>
<br>

Eduardo and I were eating some chocolate bites while making this, we decided to see what the prediction would be with some partially made up data. They were Meiji Chocorooms, we tested our model trained on all available data and it predicted that balanced was false, this was correct! There was very high %DV sugar and saturated fat so it was not balanced. This was not as challenging as correctly identifying a balanced recipe so we can give it an example like that as well. We passed in a Spinash Quiche recipe. The butter, heavy cream and cheese put this recipe well over the fat daily value. That makes this recipe not balanced but either prediction would seem reasonable. The model predicted False again, which was correct. This is an excellent example of why our model might struggle when even a human can't decide if the recipe seems balanced. Binary classification in this case always has a correct answer by definition but when using the model we need to remember that theres always a spectrum in reality of how balanced a recipe is.

<br>

### Base Model Breakdown

After trying many models and many hyperparamters we ended with a Random Forest Classifier using 50 trees and gini. We standardized three quantitative variables: `'minutes'`, `'n_steps'`, and `'n_ingredients'`. We had one categorical variable `'ingredients'` which we used an altered multilabelBinarizer to OneHotEncode each unique ingredient which essentially becomes its own nominal boolean variable. That encoding was tricky because we had to account for there being different pools of ingredients at the time of model fitting on the training data and when transforming to testing data. This also created thousands of columns (almost 10K from training) which using a variance selector we narrowed down to ones with a variance higher than the threshold therefore having an ingredient not too popular or unpopular.

We got two types of results: very low precision with decent recall and accuracy, and mediocre precision with abismal recall and mediocre accuracy. Because the worst mistake we can make is to recommend someone an unbalanced recipe and tell them it is balanced, we are prioritizing precision and tuned or models to optimize that. This came at the expense of only capturing a small portion of all the balanced recipes in our dataset therefore missing out on a lot of healthy recommendations. It is hard to say if there are any patterns in the ones we did or did not capture but that would be cool to explore. Given the 4 input variables (3 considering n_ingredients could be engineered) did not have a ton of information that clearly or directly relates to the GDP from a theoretical perspective I believe the results are somewhat good. Ingredients is almost a great feature, but without knowing the amount of each it is not as valuable. In practice the precisoin is not good enough to be used to make actual recommendations, too many would be incorrect; much work is to be done on our next model! We will go back to the feature engineering stage to see if we can add more to increase precision to a useable and trustworthy level. If our final model still has a very poor precision we will have to think about if sacrificing some amount of precision for a huge gain in recall is worth it, that would mean having a much higher count of balanced predictions with slightly less confidence.

<br>
<br>

<!-- example viz code
 <iframe src="assets/visualization_3.html" width=700 height=500 frameBorder=0></iframe> -->

<br>
<br>

# Final Model

For our final model we decided to add three specific features. First, the quarter in which the recipe was submitted. We believe that the quarter of submission is a good way of capturing the trends of balanced and unbalanced food consumption during the year. It could be the case that more recipes in quarter 1 tend to be balanced since people are motivated at the beginning of the year.

We also though of adding description length. We decided this because usually recipes with longer description tend to be more elaborate which most of the time tend to be more unbalanced compared to simple recipes that don't require that much description. Think about it, you can it's natural to give a much longer description to a cookie sundae than to a plate of chicken and rice.

The third one was the tag column which we applied the same process as we did for the ingredients column. We thought it was important to add this column because some of the tags were like `'dietary'` that could cluster these recipes into groups that might be balanced or unbalanced.

In general adding these features helped us capture more information of the data generating process that we consider helpful for our model. Another technique we used was OneHotEncoding and PCA. We used these techniques because OHE helped us get information from a list of strings while PCA summarized all this information into fewer columns. Reducing the dimensionality of our data improved a lot our model in all the aspects.

We decided to keep using RandomForestClassifier() as our model based on the previous experimentation from the baseline model section.

> A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.

Source: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html" target="_blank">
scikit learn</a>

 We chose to use the following hyperparameters:

 PCA(n_components = 8)

 RandomForestClassifier(n_estimators = 32, criterion = 'gini', class_weight='balanced')
 
 We got these values from several grid searches that we performed to optimize the precision of our predictions. It's important to remember that we optimized for precision since we believe that recommending a recipe as balanced when it actually is unbalanced can mislead to several health issues in the future. Our final model's precision is 0.50% while our baseline model's was 0.45%. This might not seem too much, but with the data we had available we consider that it's a really good precision. The trade off was that every time we increased our precision, our recall went down, but the score was usually constant around 89%.

<br>
<br>

# Fairness Analysis

For our fairness analysis we are going to analyze the `precision parity` of the recipes with the `'occasion'` tag and the ones without it. Therefore, our Group X is recipes with the `'occasion'` tag, and Group Y is the recipes without the `'occasion'` tag, all these coming from the X_test data.

Our hyposthesis for this analysis are the following:

- **Null Hypothesis**: The classifier's precision is the same for recipes for occasions and not, and any differences are due to chance.
- **Alternative Hypothesis**: The classifier's precision is different for occasions and not.
- **Test statistic**: Absolute difference in precision (between Group X and Group Y).
- **Significance level**: 0.05

For the results we got a p-value of 0.70. In conclusion, we fail to reject the null hypothesis. Therefore, we can't say our model is unfair between the recipes with `'occasion'` tag and the ones without it.

<br>
<br>
