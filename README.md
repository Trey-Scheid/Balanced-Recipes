<!-- #region -->
<img src="https://www.fda.gov/files/20210122_NFL_MyPlate_701x363_0.png" alt="examplenorating" height=576 style="display: block; margin: 0 auto">
<!-- # The Recipe for Success: Predicting Balanced Meals
For UCSD class DSC80 project 5 -->
<br>

# Introduction

> Our exploratory data analysis (and a fun hypothesis test) on this dataset can be found <a href="https://trey-scheid.github.io/Recipe-Healthyness-Trends-by-season/" target="_blank">here</a>.

In this notebook we are going to be analyzing a dataset that contains recipes and ratings from <a href="https://www.food.com/" target="_blank">Food.com</a>. In <a href="https://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19c.pdf" target="_blank">this paper</a>, Bodhisattwa Prasad Majumder\*, Shuyang Li\*, Jianmo Ni, and Julian McAuley scraped Food.com for approximately 180K recipes and 700K interactions[^1]. Our University professors have given us a subset; the combined dataset before cleaning consists of `234,429` interactions on known recipes.

In <a href="https://trey-scheid.github.io/Recipe-Healthyness-Trends-by-season/" target="_blank">our previous project</a> which we are continuing off of, we were interested in answering this question: Do people eat more unbalanced food during the winter holiday season? And we did not find any significant results that they did. Now we are moving on to a prediction problem:

**Can we predict if a recipe is balanced (according to the FDA) without knowing its nutrition information?**

[^1]: <a href="https://aclanthology.org/D19-1613" target="_blank">Generating Personalized Recipes from Historical User Preferences</a> (Majumder et al., EMNLP-IJCNLP 2019)


<br>



<iframe src="assets/visualization_3.html" width=700 height=500 frameBorder=0></iframe>


<br>

> Under the new guidelines, something that is “healthy” needs to have the equivalent of a serving of fruits, vegetables, grains, dairy or protein foods as indicated in the Dietary Guidelines for Americans. Raw, whole fruits and vegetables automatically can bear the claim. There is a scale for different kinds of prepared products that has a nutrient requirement and percentage limits for the recommended daily intakes of added sugars, sodium and saturated fats.  

This definition, which has been six years in the making, is a comprehensive and multifaceted way to handle the label claim, which many consumers would readily accept as an indicator that a food item is unequivocally good for them.

"FDA regulates about 78 percent of the U.S. food supply. This includes everything we eat except for meat, poultry, and some egg products." -FDA https://www.fda.gov/about-fda/fda-basics/fact-sheet-fda-glance

Now that we have our data cleaned, we can add a column that classifies our recipes as `'balanced'` or `'unbalanced'`. According to the FDA, meals should contain between `5%` to `20%` of the percentage of daily value (PDV). We are going to use this fact to classify our recipes in `'balanced'` and `'unbalanced'` categories.

Source: https://www.fda.gov/food/new-nutrition-facts-label/lows-and-highs-percent-daily-value-new-nutrition-facts-label

We will alter it slightly to better represent what is generally percieved as healthy. That means less than 5% sugar or fat is good, and greater than 20% protein is also alright!

<br>


USDA regulates the rest, they have their own rules for healthy items: https://www.fooddive.com/news/usda-sets-parameters-for-items-labeled-healthy/574387/


That concludes the code from our previous project, now we will do some additional cleaning and exploring that is more pertinent to our new goals with predicting balance.


We have seen that many ingredients are written in their plural and singular forms. This underrepresents their prescence in recipes since their count is distributed as two different ingredients. Therefore, to fix this we decided to strip all letters "s" from the end of the ingredients.

<br>

Here are the first few rows of the cleaned dataframe[^3]:


<iframe src="assets/sdatahead.html" width=900 height=210 frameBorder=0 title="cleaned dataset preview"></iframe>

[^3]: We only show a preview of some columns for the sake of space and formatting when they are not essential to our understanding of the dataset or the analysis.

<br>

Remember, our guiding question is:

Can we predict if a recipe is balanced (according to the FDA) without knowing its nutrition information?

This is a binary classification problem since our response variable has only two outputs True or False; a recipe is either balanced or not balanced. The EDA defines balanced recipes as ones that have nutrition info from 5-20% Daily Value. 

We think it is more important that we don't incorrectly classify a recipe as balanced when it is not, this might lead to recommendations to people for healthy recipes that are not actually healthy! These cases we want to avoid are false positives so we want to maximize precision. Some other metrics such as accuracy are missing a level of detail while the rest are including things that aren't critical such as false negatives in confusion matrices and F1 scores. 

We will not be using any interactions (`'review'` or `'avg_rating'`) on the recipes since these are posted onto the recipe after they are posted to Food.com but we would like to classify before users try the recipes!

<br>
<br>
<br>

# Baseline Model

We will choose some features that we think apply best to predicting balance of a recipe, then train a few models using the default hyperparameters. Once we see their precisions (calculated using a test dataset we set aside) we will tune the best one or two models, depending on how similar their scores were, and try to train the best model possible. Tuning will involve cross validation with a randomized/grid searches of possible combinations of hyperparameters. Finally we will have a fully trained model on the full dataset you can try on your own recipes (or just new ones scraped). 


First we need to select our features, to start we will remove columns that are ineligible according to the data generating process.

Our model should be used to predict the balanced label as true or false given information about the recipe except the nutrition. We are assuming we won't have any `'review'`s on it (under different assumpotions you could create a model for that as well). This allows us to prepare for the person uploading can be someone new to Food.com or not. If our model trained on previous `'contributor_id'`'s this may help if existing users post new recipes but not for general cases and new users, therefore we will overfit! The `'nutrition'` info itself is equivalent to creating balanced so that must also be removed. 

<br>
<br>

`'Name'`, `'submitted'`, `'steps'`, and `'description'` all cannot be directly used in the model because they will lead to overfitting, they are generally too unique to each recipe but we may be able to create features using them!

We will create these features using the sci-kit learn package transformers and drop the remaining columns in the pipeline!


We need to remember that if a simple solution works as well as a complicated one the simple solution is preferred. Thats why we will start with few features and slowly add more checking for improved performance.

<br>
<br>

For the base model we will start simple.

standardize these quantitative variables:
- minutes
- n_steps
- n_ingredients

One hot encode the values in the ingredients list.

<br>
<br>

We will remove the recipes missing a description because there are so few and it is simpler. Text imputation or deciding later when a feature is created may be more complicated and this saves time.


Currently, about 5 percent of all packaged foods are labeled “healthy,” according to the agency.
This is a severe imbalance in classes. Our performance measure: precision should not be as affected by this like accuracy would, however we still need to address it. There are two ways to deal with the imbalance. One way is to use a model that naturally deals with the imbalance, the other is to fix our training data to have equal sized classes through under or oversampling methods.
Lets create a preprocessing pipeline then try both.

<br>


There are many hyperparameters to tune between the preprocessing pipeline and the Random Forest. Due to time constraints we will perform the searches separately but ideally you would run a grid search checking all the combinations. We also will only use 3 cross-validation folds to limit the amount of training being done.

<br>


These results are a little dissapointing, but we are only using 4 variables as input! We will go back to the feature engineering stage to see if we can add more to increase precision to a useable and trustworthy level. If our final model still has a very poor precision we will have to think about if sacrificing some amount of precision for a huge gain in recall is worth it, that would mean having a much higher count of balanced predictions with slightly less confidence.
Eduardo and I were eating some chocolate bites while making this, lets see what the prediction would be with some partially made up data.

<br>

This is correct! There was very high %DV sugar and saturated fat so it was not balanced. This was not as challenging as correctly identifying a balanced recipe so we can give it an example like that as well.

<br>
<br>
<br>

## Base Model Breakdown

After trying many models and many hyperparamters we ended with a Random Forest Classifier using 50 trees and gini. We standardized three quantitative variables: 'minutes', 'n_steps', and 'n_ingredients'. We had one categorical variable 'ingredients' which we used an altered multilabelBinarizer to OneHotEncode each unique ingredient which essentially becomes its own nominal boolean variable. That encoding was tricky because we had to account for there being different pools of ingredients at the time of model fitting on the training data and when transforming to testing data. This also created thousands of columns (almost 10K from training) which using a variance selector we narrowed down to ones ___.
We got two types of results: very low precision with decent recall and accuracy, and mediocre precision with abismal recall and mediocre accuracy. Because the worst mistake we can make is to recommend someone an unbalanced recipe and tell them it is balanced, we are prioritizing precision and tuned or models to optimize that. This came at the expense of only capturing a small portion of all the balanced recipes in our dataset therefore missing out on a lot of healthy recommendations. It is hard to say if there are any patterns in the ones we did or did not capture but that would be cool to explore. Given the 4 input variables (3 considering n_ingredients could be engineered) did not have a ton of information that clearly or directly relates to the GDP from a theoretical perspective I believe the results are somewhat good. Ingredients is almost a great feature, but without knowing the amount of each it is not as valuable. In practice the precisoin is not good enough to be used to make actual recommendations, too many would be incorrect; much work is to be done on our next model!


<br>
<br>

Model and Feature Breakdown:

Standardize (quantitative):
- `'minutes'`
- `'n_steps'`
- `'n_ingredients'`

BOW and PCA (categorical):
- `'ingredients'`
- `'tags'`

New Features:
- quarter of year <-- `'submitted'` month in _ quartile
- length of description

<br>

Other feature ideas:

- time of day based on submitted

BOW + PCA, or word2vec
- `'description'`
- `'steps'`






[^1]: <a href="https://aclanthology.org/D19-1613" target="_blank">Generating Personalized Recipes from Historical User Preferences</a> (Majumder et al., EMNLP-IJCNLP 2019)








