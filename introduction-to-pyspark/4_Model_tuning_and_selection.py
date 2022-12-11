# What is logistic regression?
'''
The model you'll be fitting in this chapter is called a logistic regression. This model is very similar to a linear regression, but instead of predicting a numeric variable, it predicts the probability (between 0 and 1) of an event.

To use this as a classification algorithm, all you have to do is assign a cutoff point to these probabilities. If the predicted probability is above the cutoff point, you classify that observation as a 'yes' (in this case, the flight being late), if it's below, you classify it as a 'no'!

You'll tune this model by testing different values for several hyperparameters. A hyperparameter is just a value in the model that's not estimated from the data, but rather is supplied by the user to maximize performance. For this course it's not necessary to understand the mathematics behind all of these values - what's important is that you'll try out a few different choices and pick the best one.

Why do you supply hyperparameters?
They explain information about the data.
They improve model performance. √
They improve model fitting speed.
'''


# Create the modeler
'''
The Estimator you'll be using is a LogisticRegression from the pyspark.ml.classification submodule.

Instructions
100 XP
Import the LogisticRegression class from pyspark.ml.classification.
Create a LogisticRegression called lr by calling LogisticRegression() with no arguments.
'''

# Import LogisticRegression
from pyspark.ml.classification import LogisticRegression

# Create a LogisticRegression Estimator
lr = LogisticRegression()


# Cross validation
# two hyperparameters, elasticNetParam and regParam
'''
In the next few exercises you'll be tuning your logistic regression model using a procedure called k-fold cross validation. This is a method of estimating the model's performance on unseen data (like your test DataFrame).

It works by splitting the training data into a few different partitions. The exact number is up to you, but in this course you'll be using PySpark's default value of three. Once the data is split up, one of the partitions is set aside, and the model is fit to the others. Then the error is measured against the held out partition. This is repeated for each of the partitions, so that every block of data is held out and used as a test set exactly once. Then the error on each of the partitions is averaged. This is called the cross validation error of the model, and is a good estimate of the actual error on the held out data.

You'll be using cross validation to choose the hyperparameters by creating a grid of the possible pairs of values for the two hyperparameters, elasticNetParam and regParam, and using the cross validation error to compare all the different models so you can choose the best one!

What does cross validation allow you to estimate?

The model's error on held out data.  √
The model's error on data used for fitting.
The time it will take to fit the model.
'''


# Create the evaluator
'''
The first thing you need when doing cross validation for model selection is a way to compare different models. Luckily, the pyspark.ml.evaluation submodule has classes for evaluating different kinds of models. Your model is a binary classification model, so you'll be using the BinaryClassificationEvaluator from the pyspark.ml.evaluation module.

This evaluator calculates the area under the ROC. This is a metric that combines the two kinds of errors a binary classifier can make (false positives and false negatives) into a simple number. You'll learn more about this towards the end of the chapter!

Instructions
Import the submodule pyspark.ml.evaluation as evals.
Create evaluator by calling evals.BinaryClassificationEvaluator() with the argument metricName="areaUnderROC".
'''

# Import the evaluation submodule
import pyspark.ml.evaluation as evals

# Create a BinaryClassificationEvaluator
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")


# Make a grid
'''
Next, you need to create a grid of values to search over when looking for the optimal hyperparameters. The submodule pyspark.ml.tuning includes a class called ParamGridBuilder that does just that (maybe you're starting to notice a pattern here; PySpark has a submodule for just about everything!).

You'll need to use the .addGrid() and .build() methods to create a grid that you can use for cross validation. The .addGrid() method takes a model parameter (an attribute of the model Estimator, lr, that you created a few exercises ago) and a list of values that you want to try. The .build() method takes no arguments, it just returns the grid that you'll use later.

Instructions
Import the submodule pyspark.ml.tuning under the alias tune.
Call the class constructor ParamGridBuilder() with no arguments. Save this as grid.
Call the .addGrid() method on grid with lr.regParam as the first argument and np.arange(0, .1, .01) as the second argument. This second call is a function from the numpy module (imported as np) that creates a list of numbers from 0 to .1, incrementing by .01. Overwrite grid with the result.
Update grid again by calling the .addGrid() method a second time create a grid for lr.elasticNetParam that includes only the values [0, 1].
Call the .build() method on grid and overwrite it with the output.
'''

# Import the tuning submodule
import pyspark.ml.tuning as tune

# Create the parameter grid
grid = tune.ParamGridBuilder()

# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])

# Build the grid
grid = grid.build()


# Make the validator
'''
The submodule pyspark.ml.tuning also has a class called CrossValidator for performing cross validation. This Estimator takes the modeler you want to fit, the grid of hyperparameters you created, and the evaluator you want to use to compare your models.

The submodule pyspark.ml.tune has already been imported as tune. You'll create the CrossValidator by passing it the logistic regression Estimator lr, the parameter grid, and the evaluator you created in the previous exercises.

Instructions
Create a CrossValidator by calling tune.CrossValidator() with the arguments:
estimator=lr
estimatorParamMaps=grid
evaluator=evaluator
Name this object cv.
'''

# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr,
               estimatorParamMaps=grid,
               evaluator=evaluator
               )
               


# Fit the model(s)
# fit
'''
You're finally ready to fit the models and select the best one!

Unfortunately, cross validation is a very computationally intensive procedure. Fitting all the models would take too long on DataCamp.

To do this locally you would use the code:

# Fit cross validation models
models = cv.fit(training)

# Extract the best model
best_lr = models.bestModel

Remember, the training data is called training and you're using lr to fit a logistic regression model. Cross validation selected the parameter values regParam=0 and elasticNetParam=0 as being the best. These are the default values, so you don't need to do anything else with lr before fitting the model.

Instructions
Create best_lr by calling lr.fit() on the training data.
Print best_lr to verify that it's an object of the LogisticRegressionModel class.
'''

# Call lr.fit()
best_lr = lr.fit(training)

# Print best_lr
print(best_lr)

# LogisticRegressionModel: uid=LogisticRegression_8042c797b8f4, numClasses=2, numFeatures=83


# Evaluating binary classifiers
'''
For this course we'll be using a common metric for binary classification algorithms call the AUC, or area under the curve. In this case, the curve is the ROC, or receiver operating curve. The details of what these things actually measure isn't important for this course. All you need to know is that for our purposes, the closer the AUC is to one (1), the better the model is!

If you've created a perfect binary classification model, what would the AUC be?
1
'''


# Evaluate the model
'''
Remember the test data that you set aside waaaaaay back in chapter 3? It's finally time to test your model on it! You can use the same evaluator you made to fit the model.

Instructions
Use your model to generate predictions by applying best_lr.transform() to the test data. Save this as test_results.
Call evaluator.evaluate() on test_results to compute the AUC. Print the output.
'''

# Use the model to predict the test set
# transform
test_results = best_lr.transform(test)

# Evaluate the predictions
print(evaluator.evaluate(test_results))
# 0.7123313100891033


# Congratulations! What do you think of the AUC? Your model isn't half bad! You went from knowing nothing about Spark to doing advanced machine learning. Great job on making it to the end of the course! The next steps are learning how to create large scale Spark clusters and manage and submit jobs so that you can use models in the real world. Check out some of the other DataCamp courses that use Spark! And remember, Spark is still being actively developed, so there's new features coming all the time!