# AmazonReviewsDataPrediction
The purpose of the project is the predict the rating of an Amazon product based on reviews.

Top 32 out of 170 in a kaggle competition https://www.kaggle.com/competitions/cs-506-midterm-a1-b1/leaderboard


**File descriptions:**

train.csv - 1,697,533 unique reviews from Amazon Movie Reviews, with their associated star ratings and metadata. It is not necessary to use all reviews, or metadata for training. Some reviews will be missing a value in the 'Score' column. That is because, these are the scores you want to predict.
test.csv - Contains a table with 300,000 unique reviews. The format of the table has two columns; i) 'Id': contains an id that corresponds to a review in train.csv for which you predict a score ii) 'Score': the values for this column are missing since it will include the score predictions. You are required to predict the star ratings of these Id using the metadata in train.csv.
sample.csv - a sample submission file. The 'Id' field is populated with values from test.csv. Kaggle will only evaluate submission files in this exact same format.
<br>
**Data fields:**

ProductId - unique identifier for the product
UserId - unique identifier for the user
HelpfulnessNumerator - number of users who found the review helpful
HelpfulnessDenominator - number of users who indicated whether they found the review helpful
Score - rating between 1 and 5
Time - timestamp for the review
Summary - brief summary of the review
Text - text of the review
Id - a unique identifier associated with a review

Code Instructions <br>
1- Download the train.csv and test.csv files from Kaggle into the data/ folder <br>
2-feature_extraction.py will help you to generate features as well as generate X_test.csv which is test.csv but with the features from train.csv and whatever other features you added. <br>
3-Run predict-svc.py to predict the score using SVC <br>
