# Combined notebook script
# Generated: 2025-11-21T18:56:38
# This script aggregates multiple notebooks.
# Markdown cells are included as comments.
# IPython/Jupyter magics are preserved and require IPython to run.

# Optional: set the dataset path students should use:
DATASET_PATH = 'data/your_dataset.csv'  # Update as needed

# ================================================
# NOTEBOOK SECTION
# ================================================
# Notebook: LABS-03_Systems.ipynb

# --------------------------------
# Cell 1 - markdown
# --------------------------------
# # LABS-3: Systems Project
# 
# In this notebook you will run a few cells to ensure that your environment is properly set up.

# --------------------------------
# Cell 2 - code
# --------------------------------
## First cell
# run this for step 11 in your instructions

print("Hello, World!")

# --------------------------------
# Cell 3 - code
# --------------------------------
## Second cell
# run this for step 14 in your instructions

import pandas as pd

# --------------------------------
# Cell 4 - code
# --------------------------------
## Part 3: Accessing your data
# edit this code to load your data into your workspace

data = pd.read_csv() #add your data file path to this line!
data.head()

# --------------------------------
# Cell 5 - code
# --------------------------------




# ================================================
# NOTEBOOK SECTION
# ================================================
# Notebook: LABS-06_Design-2.ipynb

# --------------------------------
# Cell 1 - markdown
# --------------------------------
# # LABS-6: Design Project
# 
# In this notebook you will run and edit the code to perform some exploratory data analytics (EDA) and to develop and answer a question using data.
# 
# **Data**\
# This dataset comes from IMDB and can be accessed on [Kaggle](https://www.kaggle.com/datasets/ashpalsingh1525/imdb-movies-dataset).

# --------------------------------
# Cell 2 - markdown
# --------------------------------
# ## Set up environment

# --------------------------------
# Cell 3 - code
# --------------------------------
## import packages

import pandas as pd #data manipulation & analysis
import numpy as np #arrays & math
import matplotlib.pyplot as plt #data visualization
import seaborn as sns #statistical visualization
from scipy.stats import gaussian_kde #scientific computing

# --------------------------------
# Cell 4 - code
# --------------------------------
# Read in data
data = pd.read_csv("imdb_movies.csv")

# --------------------------------
# Cell 5 - markdown
# --------------------------------
# ## Understand your data
# 
# First we want to understand what the data looks like and how much of it there is. \
# To do that, we can start by looking at: 
# - the shape of the data
# - viewing a portion of the data
# - checking the data types for each column
# - looking at the summary stats of our numeric columns
# 
# The data is stored in a [pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html). This is a tabular data structure that can store different types of data like numbers and text. Dataframes are an object in python, and therfore have attributes and methods that we can use to help us understand the data. You will learn several of these below.

# --------------------------------
# Cell 6 - markdown
# --------------------------------
# ### Shape
# 
# The [shape](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html#pandas.DataFrame.shape) attribute tells us the shape of the data, or how many rows and columns are in our dataset. 
# 
# Since this is an attribute, we call it using this format: `df.shape` where `df` is the name of your dataframe.
# 
# This attribute returns the shape in this format: (# rows, # columns)
# 
# Run the cell below to get the shape of the dataframe and answer **question 1**.

# --------------------------------
# Cell 7 - code
# --------------------------------
data.shape

# --------------------------------
# Cell 8 - markdown
# --------------------------------
# ### View a portion of the Data
# 
# It is useful to be able to see what our data actually looks like - but we usually only need to see a few rows to understand what it looks like. Dataframes have a method called [`.head()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html#pandas.DataFrame.head) that displays the first *n* rows of the dataframe. 
# 
# Since this is a method, we call it using this format: `df.head()` where `df` is the name of your dataframe.
# 
# You can specify how many rows you want to see by specifying the *n* parameter inside the parenthesis. Let's look at the first 10 rows of our dataframe. Run the cell below to see data.

# --------------------------------
# Cell 9 - code
# --------------------------------
data.head(n=10)

# --------------------------------
# Cell 10 - markdown
# --------------------------------
# ### Check data types for each column
# 
# Now we will check the data types so we know how to handle each column during cleaning and analysis. The [`.info()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html#pandas.DataFrame.info) method gives us a quick summary of the dataset. It tells us how many entries (rows) there are, the names of the columns, the data type stored in each column, and how many values are missing.
# 
# Since this is a method, we call it like a function using this format: `df.info()` where `df` is the name of your DataFrame.
# 
# This method prints the details directly to the workspace.
# 
# 
# **In the cell below, write the code to call the .info() method on `data`. Use the previous examples and documentation linked above to help you.**

# --------------------------------
# Cell 11 - code
# --------------------------------
# ENTER YOUR CODE TO RUN .info() HERE


# --------------------------------
# Cell 12 - markdown
# --------------------------------
# Here are a few helpful tips to help you answer **question 5**:
# - Python stores numbers either as integers (int) or floating point decimals (float). Python stores other data types (like strings and lists) as objects.
# - "non-null" means rows without missing values.

# --------------------------------
# Cell 13 - markdown
# --------------------------------
# ### Look at summary stats for numeric columns
# 
# Now we will summarize our columns to spot patterns and outliers during cleaning and analysis. The [`.describe()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html#pandas.DataFrame.describe) method gives us a quick set of summary statistics. By default, it reports the following values for numeric columns: 
# - count
# - mean
# - standard deviation
# - minimum
# - quartiles (25%, 50%, 75%)
# - maximum
# 
# Since this is a method, we call it using this format: `df.describe()` where `df` is the name of your DataFrame.
# 
# This method returns a new DataFrame of summary statistics.
# 
# **In the cell below, write the code to call the .describe() method on `data`. Use the previous examples and documentation linked above to help you.**

# --------------------------------
# Cell 14 - code
# --------------------------------
# ENTER YOUR CODE TO RUN .describe() HERE


# --------------------------------
# Cell 15 - markdown
# --------------------------------
# ## Visualize the Data
# 
# Now we want to visualize the data to see if we can pull out any initial trends off the bat. These will inform the questions we ask and our modeling process later on. We will use 3 common graphs to look at some baseline relationships and distributions of our data:
# - Scatterplot
# - Bar chart
# - Density plot
# 
# It is important to remember that the point of this process it to understand trends and distributions of your data. There are many other plots data scientists use to do this - these are just a few basic ones.
# 
# To visualize the data we will use 2 packages: [`MatPlotLib`](https://matplotlib.org/) and [`Seaborn`](https://seaborn.pydata.org/). These were imported at the beginning of the notebook with aliases - short hand that we can use to reference the package in our code. The alias for `MatPlotLib` is `plt` and `Seaborn` is `sns`. These are the standard aliases used for these packages. 

# --------------------------------
# Cell 16 - markdown
# --------------------------------
# ### Scatterplot
# 
# [Scatterplots](https://matplotlib.org/stable/gallery/shapes_and_collections/scatter.html) show the relationship between 2 or more variables. You can add other features, like point shape and color, to your graph to see relationships of additional variables.
# 
# Run the cell below to create a scatterplot of the relationship between movie budget and revenue.

# --------------------------------
# Cell 17 - code
# --------------------------------
fig, ax = plt.subplots(figsize=(15, 10))
ax.scatter(data['budget_x'], data['revenue'])

plt.ylabel("Revenue (in billions USD)")
plt.xlabel("Budget (in one hundred millions USD)")
plt.title("Budget and Revenue relationship")
plt.show()

# --------------------------------
# Cell 18 - markdown
# --------------------------------
# ### Bar Chart
# 
# A [Bar Chart](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html#matplotlib.pyplot.bar) shows comparisons between categories by using rectangular bars. The height of each bar represents the value or frequency of that category. You can customize bar color, orientation, and grouping to highlight patterns or differences across multiple variables.
# 
# The cell below has the code to produce a bar chart using the default color settings. **Use the documentation linked above to change the color of the bars.**

# --------------------------------
# Cell 19 - code
# --------------------------------
country_counts = data['country'].value_counts()

fig, ax = plt.subplots(figsize=(15, 10))
ax.bar(country_counts.index, country_counts.values) 
ax.tick_params("x", rotation=45)

plt.ylabel("count")
plt.xlabel("country")
plt.title("Count of movies per country")
plt.show()

# --------------------------------
# Cell 20 - markdown
# --------------------------------
# ### Density plot
# 
# A [Density Plot](https://seaborn.pydata.org/generated/seaborn.kdeplot.html) shows the distribution of a continuous variable by estimating its probability density function. This is similar to histograms, but density plots use smooth curves to represent data, making it easier to compare distributions and spot patterns. You can adjust the smoothness and overlay multiple curves to explore differences across groups.
# 
# The cell below has the code to produce a Density plot, but is missing a title. **Add the code to produce a title before running the cell.**

# --------------------------------
# Cell 21 - code
# --------------------------------
x = data['score']
plt.figure(figsize=(15, 10))
sns.kdeplot(data=data, x="score")

## ADD CODE TO PRODUCE TITLE HERE
plt.show()

# --------------------------------
# Cell 22 - markdown
# --------------------------------
# ## Develop and Answer a Question

# --------------------------------
# Cell 23 - code
# --------------------------------
correlation_matrix = data[['score', 'budget_x', 'revenue']].corr()

plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()



# ================================================
# NOTEBOOK SECTION
# ================================================
# Notebook: Fairness_lab.ipynb

# --------------------------------
# Cell 1 - markdown
# --------------------------------
# # Fairness in Machine Learning Example
# 
# In this notebook we will explore an example of unnmitigated machine learning to observe how a model performs for different protected classes. We will be using the Fairlearn package to calculate various metrics to help us understand how our model performs across different classes.
# 
# Check out their user guide for more information on the package! https://fairlearn.org/v0.10/user_guide/fairness_in_machine_learning.html

# --------------------------------
# Cell 2 - markdown
# --------------------------------
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UVADS/DS1001/blob/main/code/Fairness_lab.ipynb)

# --------------------------------
# Cell 3 - code
# --------------------------------
# You will have to run this cell to install the fairlearn package we'll be using today. You should only have to do this once (unless you restart your runtime).

!pip install fairlearn

# --------------------------------
# Cell 4 - code
# --------------------------------
## You may have to run this line if you get an error message when you run the cell under 'Load in and explore/clean up the data'
## If you do, remove the # on the line below and run this cell

# !pip install pandas==2.0.3

# --------------------------------
# Cell 5 - code
# --------------------------------
# import packages

import pandas as pd
import numpy as np
import fairlearn.metrics
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import count, true_positive_rate, false_positive_rate, selection_rate, demographic_parity_ratio

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) ## ignore deprecation warnings

# --------------------------------
# Cell 6 - markdown
# --------------------------------
# # Adult Census Data
# 
# In this model we will be using demographic variables from census data to predict whether someone makes >50k or <=50k using data from: https://archive.ics.uci.edu/dataset/2/adult. A truncated/precleaned version of this is accessible through the `fairlearn` package, so we will import it from there.

# --------------------------------
# Cell 7 - markdown
# --------------------------------
# ### Load in and explore/clean up the data

# --------------------------------
# Cell 8 - code
# --------------------------------
## import and view the data

from fairlearn.datasets import fetch_adult
census_raw = fetch_adult(as_frame=True)
census = census_raw.frame #this grabs the data in a pd.dataframe format


census.head() # this prints the first 5 lines so we can see the format of the data

# --------------------------------
# Cell 9 - code
# --------------------------------
## create lists of categorical/numerical columns

census_catcols = list(census.select_dtypes('category')) # categorical columns

census_numcols = list(set(census.columns) - set(census_catcols)) # numerical columns

# --------------------------------
# Cell 10 - code
# --------------------------------
## get some info on the numerical data - gives us a general idea of spread and center
census.describe().T

# --------------------------------
# Cell 11 - code
# --------------------------------
## Visualize the spread of numeric data

fig, axs = plt.subplots(3,2)
axs = axs.ravel()
for idx,ax in enumerate(axs):
    ax.hist(census[census_numcols[idx]])
    ax.set_title(census_numcols[idx])
plt.tight_layout()

# --------------------------------
# Cell 12 - code
# --------------------------------
## info on the categorical data
# This shows us the levels in the categories for the first 2 category columns

for col in census_catcols[:2]:
    print(census[col].value_counts(), "\n")

# Most of the columns have a ton of categories, we can combine some of them to collapse the categories.
# Typically we don't want ot have more than 5ish categories in a given column

# --------------------------------
# Cell 13 - code
# --------------------------------
## Collapsing some categories...

# combining similar working classes
census['workclass'].replace(['Without-pay', 'Never-worked',], 'No-inc', inplace=True)
census['workclass'].replace(['Local-gov', 'State-gov', 'Federal-gov'], 'Gov', inplace=True)
# print(census['workclass'].value_counts())

# making race binary White/Non-White
census['race'] = (census.race.apply(lambda x: x if x == 'White' else "Non-White")).astype('category')
# print(census['race'].value_counts())

# combining similar education classes
census['education'].replace(['11th', '10th', '9th', '12th',], 'Some-HS', inplace=True)
census['education'].replace(['7th-8th', '5th-6th', '1st-4th', 'Preschool',], 'No-HS', inplace=True)
census['education'].replace(['Assoc-voc', 'Assoc-acdm', 'Prof-school'], 'Continued Ed', inplace=True)
census['education'].replace(['Bachelors', 'Masters', 'Doctorate'], 'College_+', inplace=True)
# print(census['education'].value_counts())

# combining similar marital statuses
census['marital-status'].replace(['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], 'Married', inplace=True)
census['marital-status'].replace(['Divorced', 'Separated', 'Widowed'], 'Was-Married', inplace=True)
# print(census['marital-status'].value_counts())

# keeping only the top 4 countries (based on number of observations), grouping all others into "Other" category
top_country = census['native-country'].value_counts()[:5]
census['native-country'] = (census['native-country'].apply(lambda x: x if x in top_country else "Other")).astype('category')
# print(census['native-country'].value_counts())

# keeping only the top 4 occupations (based on number of observations), grouping all others into "Other" category
top_occ = census['occupation'].value_counts()[:5]
census['occupation'] = (census['occupation'].apply(lambda x: x if x in top_occ else "Other")).astype('category')
# print(census['occupation'].value_counts())

# --------------------------------
# Cell 14 - markdown
# --------------------------------
# ##### A little more pre-processing...

# --------------------------------
# Cell 15 - code
# --------------------------------
# Scale numbers, One hot encode categories

census[census_numcols] = MinMaxScaler().fit_transform(census[census_numcols]) #scale the numerical values so they are all on the same scale
census_onehot = pd.get_dummies(census, columns = census_catcols) # creates dummy variables to one-hot encode all categorical variables

## One hot encoding creates a column for each category in a feature and assigns it a True/False value.
## For example, the 'workclass' column will be broken up into a column for each category ('workclass_Gov', 'workclass_No-inc', etc).
## A government workclass observation would have a True value in the 'workclass_Gov' column and a False value in all the other workclass columns.
## This is a common strategy you'll see in machine learning - also with 1/0 values instead of True/False (respectively).

# --------------------------------
# Cell 16 - code
# --------------------------------
census_onehot.drop(['class_<=50K', 'race_White', 'sex_Male'], axis=1, inplace=True) # drop binary category duplicates
census_onehot.head() # visualize what the data looks like after being scaled/one hot encoded

# --------------------------------
# Cell 17 - markdown
# --------------------------------
# Finally, we need to split it into a training set (to build our model) and testing set (to see how it performs on data it was not trained on).
# 
# We also need to split our data into our target ("class_>50K" - denoted as y) and features (everything else - denoted as x)

# --------------------------------
# Cell 18 - code
# --------------------------------
# split the data into train and test for model

#seperate into features and target ("class_>50K")
census_x = census_onehot.loc[:, census_onehot.columns != "class_>50K"]
census_y = census_onehot.loc[:, census_onehot.columns == "class_>50K"]

#train/test split (75/25)
X_train, X_test, y_train, y_test = train_test_split(census_x, census_y, test_size=0.25, random_state=9658)

# --------------------------------
# Cell 19 - markdown
# --------------------------------
# # Now, let's look at our data and model and evaluate the fairness
# 
# You will answer the following questions using the code/output below.

# --------------------------------
# Cell 20 - markdown
# --------------------------------
# ## Questions
# 
# ### 1. The metrics we will be using in this lecture are True Positive Rate, False Positive Rate, Selection Rate, Demographic Parity Ratio, and Equalized Odds Ratio.
# 
# ### 2. What are the protected classes in this dataset? Are these classes equally represented in the data?
# 
# ### 3. For any given protected class, what group is being favored in the model?
# 
# ### 4. Based on the fairness metrics you observed, is the model fair – why/why not?

# --------------------------------
# Cell 21 - markdown
# --------------------------------
# ### Looking at the data distribution
# 
# Type the name of the protected class you'd like to explore in the quotes below. Be sure to use the exact name (case sensitive!) of the column from the data frame above.

# --------------------------------
# Cell 22 - code
# --------------------------------
protectedClass = "education" # type the protected class you'd like to explore in the quotes here

print(census[protectedClass].value_counts()) #print the number of observations in each class

#visualize the difference in class representation
plt.bar(census[protectedClass].value_counts().index.values, census[protectedClass].value_counts().values)
plt.ylabel('count')
plt.xlabel(protectedClass)
plt.title(f"Proctected Class Distribution - {protectedClass}")
plt.show()

# --------------------------------
# Cell 23 - markdown
# --------------------------------
# ## Model buliding
# 
# It's finally time to build our model!
# 
# We'll be building a simple logistic regression model to predict if a person makes more than 50k a year.
# 
# Basically, a logistic regression works by calculating a *probability* of an observation being in a specified class for the target variable. So in this case, our model will produce a probability of a person making more than 50k. This probability is compared to a threshold value, and if the probability is above the threshold is will be categorized as a positive outcome (in this case, making more thank 50k). For more information on logistic regressions, check out this IBM page: https://www.ibm.com/topics/logistic-regression#:~:text=Logistic%20regression%20estimates%20the%20probability,given%20dataset%20of%20independent%20variables.

# --------------------------------
# Cell 24 - code
# --------------------------------
## train model

lreg = LogisticRegression() #initialize a logistic regression model
lreg.fit(X_train, y_train) #train this model using our training data

y_pred = lreg.predict(X_test) # store predicted values for the test set

# --------------------------------
# Cell 25 - markdown
# --------------------------------
# #### Average accuracy on test data

# --------------------------------
# Cell 26 - code
# --------------------------------
print("Average accuracy on test data:\t",round(lreg.score(X_test, y_test)*100,2),"%")

# --------------------------------
# Cell 27 - markdown
# --------------------------------
# ### Fairness Metrics
# 
# We are using the Fairlearn package in Python.
# 
# You will need to understand what the metric used below mean and how they are calculated. You can find information on the functions used in their documentation: https://fairlearn.org/v0.10/api_reference/index.html#module-fairlearn.metrics

# --------------------------------
# Cell 28 - code
# --------------------------------
# Construct a function dictionary with the metrics we'd like for each class
my_metrics = {
    'true positive rate' : true_positive_rate,
    'false positive rate' : false_positive_rate,
    'selection rate' : selection_rate,
    'count' : count
}
# Construct a MetricFrame for race
mf_race = MetricFrame(
    metrics=my_metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=X_test["race_Non-White"]
)

# Construct a MetricFrame for sex
mf_sex = MetricFrame(
    metrics=my_metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=X_test["sex_Female"]
)

# --------------------------------
# Cell 29 - code
# --------------------------------
# Display the by_group breakdown for race
print("Metrics by Race:")
print(mf_race.by_group)

# --------------------------------
# Cell 30 - code
# --------------------------------
def create_confmatrix(y_test, y_pred):
    '''
    creates a confusion matrix with more descriptive formatting
    '''
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel() # grab the individual values

    # create a data frame with the values in the correct spots
    conf_matrix = pd.DataFrame({'predicted positive': [tp, fp],
                                'predicted negative': [fn, tn]},
                                index=['actual positive','actual negative'])

    # return the dataframe to be saved/viewed
    return conf_matrix

# --------------------------------
# Cell 31 - markdown
# --------------------------------
# #### Overall metrics
# 
# Here is the confusion matrix for the model overall with *counts* for the true positive, false positive, true negative, and false negative.
# 
# For more information on confusion matrices, check out the wiki page: https://en.wikipedia.org/wiki/Confusion_matrix

# --------------------------------
# Cell 32 - code
# --------------------------------
# overall confusion matrix
print("Confusion matrix for all test data:")
create_confmatrix(y_test, y_pred)

# --------------------------------
# Cell 33 - markdown
# --------------------------------
# This code works because it's displaying the overall (aggregate) fairness metrics from the `MetricFrame` object `mf_race` in a formatted DataFrame.
# 
# Here's why it works:
# 
# 1. **`mf_race.overall`**: This attribute contains a dictionary-like object with the overall metrics (averaged across all groups) that were calculated when the `MetricFrame` was created. These metrics include:
#     - true positive rate
#     - false positive rate
#     - selection rate
#     - count
# 
# 2. **`pd.DataFrame(..., columns = ["overall"])`**: Converts the metrics dictionary into a pandas DataFrame with a single column named "overall"
# 
# 3. **`.T`**: Transposes the DataFrame so that instead of having metrics as rows, they become columns, making it easier to read and compare with the by-group breakdowns shown in later cells
# 
# The result is a clean, single-row DataFrame showing the model's performance metrics aggregated across all test data, which serves as a baseline for comparison when you look at how the model performs for different protected classes (race and sex) in the cells below.
# 
# This is particularly useful in fairness analysis because you can compare these overall metrics to the by-group metrics (like `mf_race.by_group`) to identify disparities in model performance across different demographic groups.

# --------------------------------
# Cell 34 - code
# --------------------------------
## The overall metrics. You'll use these to compare to with the metrics broken down by each protected class below.
## Think about how the differing performance would impact that group based on your understanding of each metric.

# --------------------------------
# Cell 35 - markdown
# --------------------------------
# Now we can look at fairness metrics for each protected class:

# --------------------------------
# Cell 36 - markdown
# --------------------------------
# #### Race

# --------------------------------
# Cell 37 - code
# --------------------------------
## metrics broken down by race classes. Compare these to the metrics above.
mf_race.by_group

# --------------------------------
# Cell 38 - markdown
# --------------------------------
# Definitions
# 
# 1. Demographic Parity Ratio: Measures whether different groups (Male vs Female) are selected at similar rates by the model. A ratio of 1.0 means perfect parity - both groups are predicted to earn >50K at the same rate. Values below 1.0 indicate one group is favored over another. For example, 0.5 means one group is selected at half the rate of the other.
# 2. Equality of Odds Ratio: The equalized odds ratio of 1 means that all groups have the same true positive, true negative, false positive, and false negative rates

# --------------------------------
# Cell 39 - code
# --------------------------------
# Derived fairness metrics. Be sure you understand the scale and meaning of these.

dpr_race = fairlearn.metrics.demographic_parity_ratio(y_test, y_pred, sensitive_features=X_test.filter(regex="race.*"))
print("Demographic Parity ratio:\t", dpr_race)

eodds_race = fairlearn.metrics.equalized_odds_ratio(y_test, y_pred, sensitive_features=X_test.filter(regex="race.*"))
print("Equalized Odds ratio:\t\t", eodds_race)

# --------------------------------
# Cell 40 - markdown
# --------------------------------
# #### Sex

# --------------------------------
# Cell 41 - code
# --------------------------------
## metrics broken down by sex classes. Compare these to the metrics above.

mf_sex.by_group

# --------------------------------
# Cell 42 - code
# --------------------------------
# Derived fairness metrics. Be sure you understand the scale and meaning of these.

dpr_sex = fairlearn.metrics.demographic_parity_ratio(y_test, y_pred, sensitive_features=X_test.filter(regex="sex.*"))
print("Demographic Parity ratio:\t", dpr_sex)

eodds_sex = fairlearn.metrics.equalized_odds_ratio(y_test, y_pred, sensitive_features=X_test.filter(regex="sex.*"))
print("Equalized Odds ratio:\t\t", eodds_sex)

# --------------------------------
# Cell 43 - code
# --------------------------------
# Let's break down the demographic parity ratio for sex with detailed visualizations

# First, let's see the selection rates for each sex group
print("=" * 60)
print("DEMOGRAPHIC PARITY RATIO EXPLANATION")
print("=" * 60)
print("\nThe Demographic Parity Ratio compares the selection rates")
print("between different groups (Male vs Female).")
print("\nFormula: min(rate_group1, rate_group2) / max(rate_group1, rate_group2)")
print("A value of 1.0 = perfect parity (both groups selected equally)")
print("A value < 1.0 = disparity exists (one group favored over another)")
print("=" * 60)

# Calculate selection rates for each sex group
male_mask = X_test["sex_Female"] == False
female_mask = X_test["sex_Female"] == True

male_selection_rate = y_pred[male_mask].sum() / len(y_pred[male_mask])
female_selection_rate = y_pred[female_mask].sum() / len(y_pred[female_mask])

print(f"\n📊 SELECTION RATES BY SEX:")
print(f"Male (predicted >50K):   {male_selection_rate:.4f} ({male_selection_rate*100:.2f}%)")
print(f"Female (predicted >50K): {female_selection_rate:.4f} ({female_selection_rate*100:.2f}%)")
print(f"\nDemographic Parity Ratio: {dpr_sex:.4f}")
print(f"Interpretation: Females are selected at {dpr_sex*100:.2f}% the rate of males")



# --------------------------------
# Cell 44 - code
# --------------------------------
# Visualization 1: Selection rates comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))


# Plot 1: Selection rates bar chart
categories = ['Male', 'Female']
selection_rates = [male_selection_rate, female_selection_rate]
colors = ['#3498db', '#e74c3c']

axes[0].bar(categories, selection_rates, color=colors, alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Selection Rate (Predicted >50K)', fontsize=11)
axes[0].set_title('Selection Rates by Sex\n(What % are predicted to earn >50K?)', fontsize=12, fontweight='bold')
axes[0].set_ylim(0, max(selection_rates) * 1.2)
for i, v in enumerate(selection_rates):
    axes[0].text(i, v + 0.01, f'{v*100:.1f}%', ha='center', fontweight='bold')
axes[0].axhline(y=male_selection_rate, color='gray', linestyle='--', alpha=0.5, label='Male rate (reference)')
axes[0].legend()

# Plot 2: Counts of predictions
male_positive = y_pred[male_mask].sum()
male_negative = (~y_pred[male_mask]).sum()
female_positive = y_pred[female_mask].sum()
female_negative = (~y_pred[female_mask]).sum()

x = np.arange(2)
width = 0.35

axes[1].bar(x - width/2, [male_positive, female_positive], width, label='Predicted >50K', color='#2ecc71', alpha=0.8)
axes[1].bar(x + width/2, [male_negative, female_negative], width, label='Predicted ≤50K', color='#95a5a6', alpha=0.8)
axes[1].set_ylabel('Count of Predictions', fontsize=11)
axes[1].set_title('Distribution of Predictions by Sex', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(categories)
axes[1].legend()

# Add count labels
for i, (pos, neg) in enumerate([(male_positive, male_negative), (female_positive, female_negative)]):
    axes[1].text(i - width/2, pos + 50, str(pos), ha='center', fontweight='bold', fontsize=9)
    axes[1].text(i + width/2, neg + 50, str(neg), ha='center', fontweight='bold', fontsize=9)

# Plot 3: Visual representation of disparity
axes[2].barh(['Female', 'Male'], [female_selection_rate, male_selection_rate],
             color=['#e74c3c', '#3498db'], alpha=0.7, edgecolor='black')
axes[2].set_xlabel('Selection Rate', fontsize=11)
axes[2].set_title(f'Disparity Visualization\nDPR = {dpr_sex:.4f}', fontsize=12, fontweight='bold')
axes[2].axvline(x=male_selection_rate, color='gray', linestyle='--', alpha=0.5, label='Male rate (reference)')

# Add arrows showing the gap
gap = male_selection_rate - female_selection_rate
axes[2].annotate('', xy=(male_selection_rate, 0), xytext=(female_selection_rate, 0),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
axes[2].text((male_selection_rate + female_selection_rate)/2, 0.05,
             f'Gap: {gap*100:.1f}%', ha='center', color='red', fontweight='bold')

plt.tight_layout()
plt.show()

# Additional statistics
print(f"\n📈 DETAILED STATISTICS:")
print(f"\nSample sizes:")
print(f"  Male:   {male_mask.sum()} observations")
print(f"  Female: {female_mask.sum()} observations")

print(f"\nPredictions breakdown:")
print(f"  Male predicted >50K:   {male_positive} ({male_selection_rate*100:.1f}%)")
print(f"  Female predicted >50K: {female_positive} ({female_selection_rate*100:.1f}%)")

print(f"\n⚠️  FAIRNESS ASSESSMENT:")
if dpr_sex >= 0.8:
    print("✓ Passes the 80% rule (commonly used fairness threshold)")
else:
    print(f"✗ Fails the 80% rule (DPR should be ≥ 0.8)")
    print(f"  Current DPR ({dpr_sex:.4f}) indicates significant disparity")
    print(f"  Females are underrepresented in positive predictions")

# --------------------------------
# Cell 45 - code
# --------------------------------
# Equalized Odds breakdown for sex (similar style to DPR breakdown above)

print("=" * 60)
print("EQUALIZED ODDS RATIO EXPLANATION")
print("=" * 60)
print("Equalized Odds requires similar error and success rates across groups.")
print("We compare True Positive Rate (TPR), False Positive Rate (FPR),")
print("True Negative Rate (TNR), and False Negative Rate (FNR) for Male vs Female.")
print(f"Equalized Odds Ratio (overall): {eodds_sex:.4f}")
print("A value of 1.0 indicates parity across all these rates.\n")

y_true = y_test["class_>50K"]

def group_confusion(mask):
    tp = np.sum((y_pred == True)  & (y_true == True)  & mask)
    fp = np.sum((y_pred == True)  & (y_true == False) & mask)
    tn = np.sum((y_pred == False) & (y_true == False) & mask)
    fn = np.sum((y_pred == False) & (y_true == True)  & mask)
    return tp, fp, tn, fn

male_tp, male_fp, male_tn, male_fn = group_confusion(male_mask)
female_tp, female_fp, female_tn, female_fn = group_confusion(female_mask)

def rates(tp, fp, tn, fn):
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan
    return tpr, fpr, tnr, fnr

male_tpr, male_fpr, male_tnr, male_fnr = rates(male_tp, male_fp, male_tn, male_fn)
female_tpr, female_fpr, female_tnr, female_fnr = rates(female_tp, female_fp, female_tn, female_fn)

print("Group performance rates:")
print(f"Male   -> TPR:{male_tpr:.3f} FPR:{male_fpr:.3f} TNR:{male_tnr:.3f} FNR:{male_fnr:.3f}")
print(f"Female -> TPR:{female_tpr:.3f} FPR:{female_fpr:.3f} TNR:{female_tnr:.3f} FNR:{female_fnr:.3f}\n")

# Ratios (smaller / larger for each metric)
def ratio(a, b):
    return min(a, b) / max(a, b) if (a > 0 and b > 0) else np.nan

tpr_ratio = ratio(male_tpr, female_tpr)
fpr_ratio = ratio(male_fpr, female_fpr)
tnr_ratio = ratio(male_tnr, female_tnr)
fnr_ratio = ratio(male_fnr, female_fnr)

print("Per-metric parity ratios (min/max):")
print(f"TPR ratio: {tpr_ratio:.3f}")
print(f"FPR ratio: {fpr_ratio:.3f}")
print(f"TNR ratio: {tnr_ratio:.3f}")
print(f"FNR ratio: {fnr_ratio:.3f}")

worst_ratio = np.nanmin([tpr_ratio, fpr_ratio, tnr_ratio, fnr_ratio])
print(f"\nWorst-case parity ratio (approx basis of Equalized Odds): {worst_ratio:.3f}")
print(f"Reported Equalized Odds Ratio: {eodds_sex:.4f}\n")

print("Fairness assessment:")
if eodds_sex >= 0.8:
    print("Passes a common 0.8 threshold.")
else:
    print("Fails a common 0.8 threshold; substantial disparity exists.")

# Visualization
fig2, axarr = plt.subplots(1, 3, figsize=(16,5))

# Panel 1: TPR/FPR comparison
metrics_names = ["TPR","FPR","TNR","FNR"]
male_vals = [male_tpr, male_fpr, male_tnr, male_fnr]
female_vals = [female_tpr, female_fpr, female_tnr, female_fnr]
x_pos = np.arange(len(metrics_names))
axarr[0].bar(x_pos - 0.2, male_vals, width=0.4, label="Male", color="#3498db")
axarr[0].bar(x_pos + 0.2, female_vals, width=0.4, label="Female", color="#e74c3c")
axarr[0].set_xticks(x_pos)
axarr[0].set_xticklabels(metrics_names)
axarr[0].set_ylabel("Rate")
axarr[0].set_title("Error/Success Rates by Sex")
axarr[0].legend()

# Panel 2: Confusion matrix counts per group (normalized)
def norm_counts(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    return [tp/total, fp/total, tn/total, fn/total]
male_norm = norm_counts(male_tp, male_fp, male_tn, male_fn)
female_norm = norm_counts(female_tp, female_fp, female_tn, female_fn)
labels_cm = ["TP","FP","TN","FN"]
x2 = np.arange(len(labels_cm))
axarr[1].bar(x2 - 0.2, male_norm, 0.4, label="Male", color="#3498db")
axarr[1].bar(x2 + 0.2, female_norm, 0.4, label="Female", color="#e74c3c")
axarr[1].set_xticks(x2)
axarr[1].set_xticklabels(labels_cm)
axarr[1].set_ylabel("Proportion")
axarr[1].set_title("Normalized Confusion Components")
axarr[1].legend()

# Panel 3: Parity ratios
parity_vals = [tpr_ratio, fpr_ratio, tnr_ratio, fnr_ratio]
axarr[2].bar(metrics_names, parity_vals, color="#8e44ad", alpha=0.7)
axarr[2].axhline(1.0, color="gray", linestyle="--", linewidth=1)
axarr[2].axhline(0.8, color="red", linestyle="--", linewidth=1)
axarr[2].set_ylim(0, 1.05)
axarr[2].set_ylabel("Parity Ratio (min/max)")
axarr[2].set_title(f"Per-Metric Parity Ratios\nWorst={worst_ratio:.3f} | EO={eodds_sex:.3f}")

plt.tight_layout()
plt.show()



# ================================================
# NOTEBOOK SECTION
# ================================================
# Notebook: LABS_09_Analytics(2).ipynb

# --------------------------------
# Cell 1 - markdown
# --------------------------------
# # LABS-9: Analytics Project
# 
# In this notebook you will run and edit the code to perform some data cleaning and run a basic kNN model.
# 
# > **What is kNN?**\
# k-Nearest Neighbors (kNN) is a machine learning algorithm used for classification tasks. At its core, kNN works by measuring the "distance" between data points. When you want to predict the category of a new data point, kNN looks at the 'k' closest points in your dataset (its "neighbors") and assigns the most common category among those neighbors to the new point.
# 
# > **How does kNN measure distance?**\
# kNN relies on distance metrics - commonly Euclidean distance (the straight-line distance between two points) - to find which data points are most similar. The algorithm compares all features (columns) in your dataset, so it's important that these features are numeric or converted to a format where distances can be calculated.
# 
# > **How will we use kNN here?**\
# In this notebook, we'll use kNN for classification: predicting whether a movie will receive a "high" or "low" score based on its features (like genre, country, budget, and more). By carefully cleaning and formatting our data, we ensure that kNN can measure distances and make meaningful predictions.
# 
# **Data**\
# This dataset comes from IMDB and can be accessed on [Kaggle](https://www.kaggle.com/datasets/ashpalsingh1525/imdb-movies-dataset).

# --------------------------------
# Cell 2 - markdown
# --------------------------------
# ## Set up environment

# --------------------------------
# Cell 3 - code
# --------------------------------
## import packages

import pandas as pd #data ingestion & cleaning
import numpy as np #numbers

# modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import kagglehub

# --------------------------------
# Cell 4 - code
# --------------------------------
# Download latest version
path = kagglehub.dataset_download("ashpalsingh1525/imdb-movies-dataset")

print("Path to dataset files:", path)

# --------------------------------
# Cell 5 - code
# --------------------------------
# Read in data
data = pd.read_csv('insert file path here')

# view data
data.head()

# --------------------------------
# Cell 6 - markdown
# --------------------------------
# ## Data Cleaning & Model Prep
# 
# Before building a machine learning model, it is essential to clean and format the data. Raw data often contains missing values, inconsistent formats, or irrelevant information that can negatively impact or break a model.\
# Many algorithms, including kNN, require numeric input or specificly formatted categorical data. By cleaning the data (removing or imputing missing values, converting strings to categorical variables, and creating dummy variables), we ensure that our dataset is structured in a way that the model can interpret and learn from effectively.
# 
# Proper data preparation leads to more accurate, reliable, and interpretable results.
# 
# There are many decisions that get made throughout this process and there is often no "right" answer - so documentating why you do things as you clean data is **key**.

# --------------------------------
# Cell 7 - markdown
# --------------------------------
# ### Missing Values
# 
# We saw in our design lab that some of our columns are missing values. Many models can not tolerate missing data (they will break the model), so we have to deal with these before passing the data through to our model.
# 
# We can use the [`.info()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html#pandas.DataFrame.info) method to see what columns are missing data. Run this below (look back at LABS-06 if you don't remember how).

# --------------------------------
# Cell 8 - code
# --------------------------------
# ADD THE CODE TO RUN .INFO() ON THE DATA HERE


# --------------------------------
# Cell 9 - markdown
# --------------------------------
# 2 columns are missing data: `genre` and `crew`.\
# Since we have a large data set for kNN, we can drop the relatively few rows that are missing data using .dropna()

# --------------------------------
# Cell 10 - code
# --------------------------------
## make a new df to make changes to
model_data = data.copy()

# --------------------------------
# Cell 11 - code
# --------------------------------
# drop rows containing NaN (missing) values
model_data.dropna(inplace=True)

# --------------------------------
# Cell 12 - markdown
# --------------------------------
# Now that we've dropped rows with missing values, our dataset is free of NaNs.
# >The output of the cell below (`model_data.info()`) confirms that all columns are complete and can be used to answer **question 1** regarding how much data was removed.

# --------------------------------
# Cell 13 - code
# --------------------------------
model_data.info()

# --------------------------------
# Cell 14 - markdown
# --------------------------------
# ### Look at Columns
# 
# When preparing data for kNN, it's important to look at the values in each column, especially for columns with categories (like country or genre).\
# kNN works by measuring the distance between data points to find the ones that are most similar, so if a column has too many different categories, it can make these distance calculations confusing and less useful. If some categories only appear a few times, they don't help much and can make the model less accurate.
# 
# To fix this, we can group these less common categories into a single 'other' category. This makes the data simpler and helps kNN focus on the most useful information when measuring distances.

# --------------------------------
# Cell 15 - code
# --------------------------------
data['status'].value_counts()

# --------------------------------
# Cell 16 - markdown
# --------------------------------
# The `status` column has 3 levels (3 distrinct values in the column). This works great for kNN!
# 
# Now let's take a look at `country`.

# --------------------------------
# Cell 17 - code
# --------------------------------
data['country'].value_counts()

# --------------------------------
# Cell 18 - markdown
# --------------------------------
# `country` has far too many columns to use in kNN. We need to collapse the smaller countries into a single "other" category.
# 
# In the next code cell, you will run the code to do this.

# --------------------------------
# Cell 19 - markdown
# --------------------------------
# #### Collapse the `country` column
# 
# The cell below contains the code to collapse the `country` column reassign any coutries with less occurances than the threshold to have the value 'other'.
# 
# > *Hint: Look at the variable assignment in the cell below to identify where the threshold for grouping countries is set to help answer **question 3**. Notice where this is used in the line below the does the actual collapsing.*

# --------------------------------
# Cell 20 - code
# --------------------------------
threshold = 100
model_data['country'] = model_data['country'].apply(lambda x: x if model_data['country'].value_counts()[x] > threshold else 'other')

# --------------------------------
# Cell 21 - code
# --------------------------------
# check the new counts after collapsing

model_data['country'].value_counts()

# --------------------------------
# Cell 22 - markdown
# --------------------------------
# Now the `country` column has 11 levels, including our new "other" category. This is still a little large for kNN, but we can always come back and adjust this if we find our model needs some tinkering.

# --------------------------------
# Cell 23 - markdown
# --------------------------------
# #### Look at other problematic columns
# 
# Now let's check out some of the other categorical columns that we'll use and see if we need to collapse or simplify any of them.

# --------------------------------
# Cell 24 - code
# --------------------------------
# Check values for 'genre' column

model_data['genre'].value_counts()

# --------------------------------
# Cell 25 - code
# --------------------------------
# Check values for 'orig_lang' column

model_data['orig_lang'].value_counts()

# --------------------------------
# Cell 26 - markdown
# --------------------------------
# Both the `genre` and `orig_lang` columns often contain multiple values separated by commas within a single cell (for example, "Animation, Adventure, Family" or "Spanish, Castilian"). This means that instead of just one value, each cell can list several genres or languages for a movie.
# 
# This is too complex for machine learning models like kNN, which work best when each column contains just one clear value per row. To make things simpler and easier to model, we will only keep the first value from each cell. This way, each movie will have just one genre and one language listed, making our data cleaner and better suited for the modeling we are doing.

# --------------------------------
# Cell 27 - markdown
# --------------------------------
# #### Get the first value for `genre` and `orig_lang`
# 
# 
# This process happens in three steps:
# 
# 1. **Define the function:**  
#     To simplify columns that contain lists of values, we we use a **function** - a reusable block of code that performs a specific task. Think of a function like a mini-program: you give it some input, it does something for you, and gives you back a result. Here, our function will extract just the first value from each cell to simplify the data so each row has only one value per column.
# 
# 2. **Apply the function:**  
#     We then apply this function to the `genre` and `orig_lang` columns, creating new columns called `top_genre` and `top_lang` with just the top value for each movie.
# 
# 3. **Recheck the counts:**  
#     After this step, we check how many unique values remain in these simplified columns to confirm that our data is now easier to work with.

# --------------------------------
# Cell 28 - code
# --------------------------------
# define the function to simplify the columns

def get_top_value(old_column_name, new_column_name):
    """
    Function to extract the first value from a column that contains multiple comma seperated values
    Appends a new column to the dataframe
    """

    col = list(model_data[old_column_name].values)

    top_list = []
    for item in col:
        item = str(item).split(",")
        item1 = item[0]
        top_list.append(item1)

    model_data[new_column_name] = top_list

# --------------------------------
# Cell 29 - code
# --------------------------------
# apply the function to the 'genre' and 'orig_lang' columns

get_top_value('genre', 'top_genre')
get_top_value('orig_lang', 'top_lang')

# --------------------------------
# Cell 30 - code
# --------------------------------
# Check the counts again

display(model_data['top_genre'].value_counts(),
        model_data['top_lang'].value_counts())

# --------------------------------
# Cell 31 - markdown
# --------------------------------
# The `top_genre` column now has 19 different values, which is easier for kNN to work with than the original genre data that had lots of combinations. By simplifying this column, the distance calculations in the model will have more meaning. If we notice that some genres are still very rare or if our model isn’t working well, we can always come back and group those rare genres into an "other" category to make things even simpler.
# 
# 
# For the `top_lang` column, we’ve also made things easier by keeping only the main language for each movie. But some languages only show up a few times, which can overly-complicate the distance calculations in the model. To fix this, we should group these less common languages into an "other" category, just like we did for the `country` column. This helps the most important languages have meaning and avoids over-complicating the model with the ones that don’t appear often.

# --------------------------------
# Cell 32 - markdown
# --------------------------------
# The cell below contains the code to collapse the `top_lang` column reassign any languages with less occurances than the threshold to have the value 'other'.
# 
# > *Hint: Look at the variable assignment in the cell below to identify where the threshold for grouping languages is set to help answer **question 3**. Notice where this is used in the line below the does the actual collapsing.*

# --------------------------------
# Cell 33 - code
# --------------------------------
#collapse top_lang

threshold = 10
model_data['top_lang'] = model_data['top_lang'].apply(lambda x: x if model_data['top_lang'].value_counts()[x] > threshold else 'other')

# --------------------------------
# Cell 34 - markdown
# --------------------------------
# ### Reformat columns
# 
# For kNN modeling, each column in your dataset should be in a format that the algorithm can easily understand and compare. This means:
# 
# - **Numeric columns** (like `budget_x`, `revenue`, and `date_x`) should contain numbers only, and be formatted as such so the model can measure distances between values. We also need to scale these columns so that the values are in the same range, which ensures all features contribute equally rather than letting large-range variables dominate.
# - **Categorical columns** (like `country`, `top_genre`, and `top_lang`) should be converted into a format the model can use. One common way to do this is by creating dummy variables (also called one-hot encoding). For example, if you have a column called `top_genre` with values like "Action", "Comedy", or "Drama", you make a new column for each possible genre: one column for "Action", one for "Comedy", one for "Drama", and so on. In each new column, you put a `1` (or `True`) if the movie belongs to that genre, and a `0` (or `False`) if it does not. This allows for distances between values to be calculated.
# - **The variable you are predicting** (in this case `score`) needs to be categorical, such as "high" or "low", so kNN can classify new data into one of these categories. While kNN *can* be used for regression (predicting specific values), it is best suited for classification - which is how we will be using it here.
# 
# By carefully reformatting each column, we ensure that kNN can accurately measure the "distance" between movies based on meaningful, comparable features—leading to better predictions and more reliable results.

# --------------------------------
# Cell 35 - markdown
# --------------------------------
# #### Date
# 
# The original `date_x` column contains full date information (month, day, year), but for our analysis we will simplify the data and make it easier for kNN to compare movies based on their release year. This also helps avoid unnecessary complexity from day/month differences that aren't meaningful for our predictions.
# 
# This is another place where if we find our model does not perform well, we may want to revisit this decision to only keep the year, as the month a movie gets released may have some predictive power as well.
# 
# **Step 1: Convert the `date_x` column to datetime format**  
# First, we use a pandas function to change the `date_x` column from a string (like "03/02/2023") into a pandas datetime object. This makes it easy to work with dates and extract parts like the year.
# 
# **Step 2: Extract the year from the datetime column**  
# Once the column is in datetime format, we can use pandas to pull out just the year for each movie and save it in a new column called `year`. This gives us a simple numeric value that is much easier for kNN to use when comparing movies.
# 
# By following these steps, we transform a complex date into a single, meaningful feature that helps our model focus on the most relevant information.

# --------------------------------
# Cell 36 - code
# --------------------------------
## Convert to datetime format

model_data['date_x'] = pd.to_datetime(model_data['date_x'])

# --------------------------------
# Cell 37 - code
# --------------------------------
## extract the year and save as a new column 'year'
# this will save as an integer

model_data['year'] = model_data['date_x'].dt.year

# --------------------------------
# Cell 38 - markdown
# --------------------------------
# #### Score
# 
# To use kNN for classification, we need to convert the movie scores from numbers to categories. In this case, we will use a 2-level classification: "high" or "low" scores.\
# This will allow us to predict whether a movie will receive a "high" or "low" score based on the characteristics in our dataset.
# 
# In order to convert our score from numeric to categorical, we must determine a "threshold" to seperate the "high" and "low" scores.\
# Use the density plot of the score distribution you created in LABS_06-Design to pick a threshold. Then:
#  - Enter your chosen threshold into the code cell below before running it.
#  - Provide an explanation for why you chose that threshold to answer **question 5**.
# 
# > ***Remember**: This is another situation where there is no "right answer" on what to choose.*\
# *This is a decision point where you should make an informed choice and note that you can adjust it later if your model performance warrants it.*

# --------------------------------
# Cell 39 - code
# --------------------------------
## reformat score

score_threshold = 90
model_data['score'] = model_data['score'].apply(lambda x: 'high' if model_data['score'][x] > score_threshold else 'low')
model_data.head()

# --------------------------------
# Cell 40 - markdown
# --------------------------------
# The score variable is now stored as strings of "high" or "low", but for kNN modeling, it must be converted and saved as a categorical variable because kNN groups data based on distinct labels. If the score is stored as a string, the algorithm may treat each unique text as a separate class, but converting it to a categorical type ensures consistent, clear groupings for accurate classification.

# --------------------------------
# Cell 41 - code
# --------------------------------
# reformat from string to category

model_data['score'] = model_data['score'].astype('category')

# --------------------------------
# Cell 42 - markdown
# --------------------------------
# #### Scale Numeric Columns
# 
# The [MinMaxScaler()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler) transforms all of our numeric columns to be on a scale of 0-1. This avoids any very large numbers (like `revenue` and `budget_x`) from overpowering the distance calculations and skewing the predictions.

# --------------------------------
# Cell 43 - code
# --------------------------------
# Select only numeric columns
numeric_cols = model_data.select_dtypes(include='number').columns

# Scale numeric columns
model_data[numeric_cols] = MinMaxScaler().fit_transform(model_data[numeric_cols])

# --------------------------------
# Cell 44 - code
# --------------------------------
## check the data types of each column
# note that score is now a 'category'

model_data.dtypes

# --------------------------------
# Cell 45 - markdown
# --------------------------------
# ### Drop extraneous columns
# 
# Some columns in our dataset are not useful for kNN modeling because they contain text, complex lists, or information that cannot be easily converted into numeric or categorical formats for distance calculations. For example, columns like `names`, `overview`, `crew`, and `orig_title` contain descriptive text or lists of people, which do not help the kNN algorithm calculate distances to compare movies in a meaningful way.
# 
# Additionally, for columns that were adjusted—such as `date_x`, `genre`, and `orig_lang`: we created new, simplified versions (`year`, `top_genre`, and `top_lang`) that are more suitable for kNN. After these adjustments, we drop the original columns to avoid redundancy and ensure our model only uses clean, relevant features.
# 
# By removing these extraneous columns, we streamline our dataset and focus on the features that will help kNN make accurate predictions.

# --------------------------------
# Cell 46 - code
# --------------------------------
## Drop the columns we won't use

model_data = model_data.drop(columns=['status', 'date_x', 'names', 'genre', 'overview', 'crew', 'orig_title', 'orig_lang'])
# Export cleaned data to CSV for later use
model_data.to_csv("imdb_movies_cleaned.csv", index=False)

#view the data now
model_data.head()

# --------------------------------
# Cell 47 - markdown
# --------------------------------
# ### Seperate features from target
# 
# It is best practice to separate the columns used for prediction from the column you want to predict into distinct variables. This makes your code easier to read and helps you avoid mistakes, like accidentally using the answer to help make predictions. Most machine learning tools expect you to give them features and targets separately, so organizing your data this way makes your workflow smoother and less confusing.

# --------------------------------
# Cell 48 - code
# --------------------------------
# features: all columns except 'score'
features = model_data.drop('score', axis=1)

# Target: score column
target = model_data['score']

# --------------------------------
# Cell 49 - markdown
# --------------------------------
# #### Dummy variables
# 
# Dummy variables are created by transforming each category in a column into its own separate column, where each row is marked with a 1 or 0 (coded as True or False) to indicate the presence or absence of that category. This process, called one-hot encoding, ensures that all categorical features are represented numerically, making them compatible with algorithms like kNN that rely on distance calculations.
# 
# For example, the `country` column contains values like "US", "AU", and "other", one-hot encoding will create new columns: `country_US`, `country_AU`, and `country_other`. Each row will have True in the column matching its country and False in the others. Similarly, for the `top_genre` column with values "Action", "Comedy", and "Drama", new columns `top_genre_Action`, `top_genre_Comedy`, and `top_genre_Drama` are created, with True/False indicating the genre for each movie.
# 
# By converting categorical data into dummy variables, we prepare our dataset for effective modeling and comparison.

# --------------------------------
# Cell 50 - code
# --------------------------------
## create dummy variables for the features dataframe
features = pd.get_dummies(features)

# preview the new features dataframe
features.head()

# --------------------------------
# Cell 51 - markdown
# --------------------------------
# ### Train/Test split
# Splitting the data into a **train set** and a **test set** is essential for building reliable machine learning models. The train set provides patterns and relationships for the algorithm to 'learn', while the test set is kept separate to evaluate performance on new, unseen data. This process helps identify overfitting - when an algorithm fits the training data too closely and fails to generalize - and ensures that results reflect true predictive power. Comparing accuracy on both sets allows for a confident assessment of how well the approach will work in real-world scenarios.
# 
# We use the [`train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function from scikit-learn to perform this split. This function randomly divides the dataset into training and testing subsets, making it easy to control the size of each set and ensure reproducibility.
# 
# > Use the linked documentation and the code below to help answer **question 8**.

# --------------------------------
# Cell 52 - code
# --------------------------------
# train test split
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=45)

# --------------------------------
# Cell 53 - markdown
# --------------------------------
# ## Create the model
# 
# To build our kNN model, we must create our model, then train, test, and evaluate it. These steps are explained in more detail below.
# 
# **Step 1: Create model - Initialize the classifier object**  
# We create a kNN classifier object using [`KNeighborsClassifier()`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html). This sets up the algorithm and specifies the nearest neighbors to classify each movie.
# 
# **Step 2: Train - Fit to the training data**  
# We train the kNN model by calling [`.fit()`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.fit). This step allows the classifier to learn patterns from the training data so it can make predictions.
# 
# **Step 3: Test - Predict on the testing data**  
# We use the trained model to predict the score category ("high" or "low") for the movies in our test set with [`.predict()`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.predict). This generates predictions for data the model has not seen before.
# 
# **Step 4: Evaluate - Calculate the accuracy of the testing data predictions**  
# We evaluate how well our model performed by comparing the predicted categories to the actual categories in the test set using [`accuracy_score()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html). This gives us a percentage that reflects the proportion of correct predictions.

# --------------------------------
# Cell 54 - code
# --------------------------------
# Create the model

knn = KNeighborsClassifier(n_neighbors=5)

# --------------------------------
# Cell 55 - code
# --------------------------------
# Train the model

knn.fit(features_train, target_train)

# --------------------------------
# Cell 56 - code
# --------------------------------
# Test the model

target_predicted = knn.predict(features_test)

# --------------------------------
# Cell 57 - code
# --------------------------------
# Evaluate the model

print("Accuracy:", accuracy_score(target_test, target_predicted))

# --------------------------------
# Cell 58 - markdown
# --------------------------------
# ## Your Turn: Adjust the model
# 
# You will now explore how the number of neighbors (`k`) affects your kNN model's accuracy. Follow these steps:
# 
# 1. **Change the value of `n_neighbors`** in the cell where the kNN model is created. Try at least 5 different values.
# 1. **For each k value**, run *all* model building cells:  
#     - Create the model  
#     - Train the model  
#     - Test the model  
#     - Evaluate the accuracy  
# 1. **Record the accuracy** for each k value you try.
# 1. **Choose the best k** based on your results and explain your reasoning to answer **quesiton 10**.
# 
# > *Tip: The "best" k is usually the one that gives the highest accuracy on the test set, but consider if the accuracy is stable or if the model seems to be overfitting or underfitting at certain values.*

# --------------------------------
# Cell 59 - markdown
# --------------------------------




