# Machine Learning Pipelines
'''
In the next two chapters you'll step through every stage of the machine learning pipeline, from data intake to model evaluation. Let's get to it!

At the core of the pyspark.ml module are the Transformer and Estimator classes. Almost every other class in the module behaves similarly to these two basic classes.

Transformer classes have a .transform() method that takes a DataFrame and returns a new DataFrame; usually the original one with a new column appended. For example, you might use the class Bucketizer to create discrete bins from a continuous feature or the class PCA to reduce the dimensionality of your dataset using principal component analysis.

Estimator classes all implement a .fit() method. These methods also take a DataFrame, but instead of returning another DataFrame they return a model object. This can be something like a StringIndexerModel for including categorical data saved as strings in your models, or a RandomForestModel that uses the random forest algorithm for classification or regression.

Which of the following is not true about machine learning in Spark?

Spark's algorithms give better results than other algorithms. √
Working in Spark allows you to create reproducible machine learning pipelines.
Machine learning pipelines in Spark are made up of Transformers and Estimators.
PySpark uses the pyspark.ml submodule to interface with Spark's machine learning routines.
'''

# Join the DataFrames
'''
In the next two chapters you'll be working to build a model that predicts whether or not a flight will be delayed based on the flights data we've been working with. This model will also include information about the plane that flew the route, so the first step is to join the two tables: flights and planes!

Instructions
First, rename the year column of planes to plane_year to avoid duplicate column names.
Create a new DataFrame called model_data by joining the flights table with planes using the tailnum column as the key.
'''

# Rename year column
planes = planes.withColumnRenamed("year", "plane_year")

# Join the DataFrames
model_data = flights.join(planes, on="tailnum", how="leftouter")

# print(model_data.show())

'''
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+--------------------+--------------+-----------+-------+-----+-----+---------+
|tailnum|year|month|day|dep_time|dep_delay|arr_time|arr_delay|carrier|flight|origin|dest|air_time|distance|hour|minute|plane_year|                type|  manufacturer|      model|engines|seats|speed|   engine|
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+--------------------+--------------+-----------+-------+-----+-----+---------+
| N846VA|2014|   12|  8|     658|       -7|     935|       -5|     VX|  1780|   SEA| LAX|     132|     954|   6|    58|      2011|Fixed wing multi ...|        AIRBUS|   A320-214|      2|  182|   NA|Turbo-fan|
| N559AS|2014|    1| 22|    1040|        5|    1505|        5|     AS|   851|   SEA| HNL|     360|    2677|  10|    40|      2006|Fixed wing multi ...|        BOEING|    737-890|      2|  149|   NA|Turbo-fan|
| N847VA|2014|    3|  9|    1443|       -2|    1652|        2|     VX|   755|   SEA| SFO|     111|     679|  14|    43|      2011|Fixed wing multi ...|        AIRBUS|   A320-214|      2|  182|   NA|Turbo-fan|
| N360SW|2014|    4|  9|    1705|       45|    1839|       34|     WN|   344|   PDX| SJC|      83|     569|  17|     5|      1992|Fixed wing multi ...|        BOEING|    737-3H4|      2|  149|   NA|Turbo-fan|
| N612AS|2014|    3|  9|     754|       -1|    1015|        1|     AS|   522|   SEA| BUR|     127|     937|   7|    54|      1999|Fixed wing multi ...|        BOEING|    737-790|      2|  151|   NA|Turbo-jet|
| N646SW|2014|    1| 15|    1037|        7|    1352|        2|     WN|    48|   PDX| DEN|     121|     991|  10|    37|      1997|Fixed wing multi ...|        BOEING|    737-3H4|      2|  149|   NA|Turbo-fan|
| N422WN|2014|    7|  2|     847|       42|    1041|       51|     WN|  1520|   PDX| OAK|      90|     543|   8|    47|      2002|Fixed wing multi ...|        BOEING|    737-7H4|      2|  140|   NA|Turbo-fan|
| N361VA|2014|    5| 12|    1655|       -5|    1842|      -18|     VX|   755|   SEA| SFO|      98|     679|  16|    55|      2013|Fixed wing multi ...|        AIRBUS|   A320-214|      2|  182|   NA|Turbo-fan|
| N309AS|2014|    4| 19|    1236|       -4|    1508|       -7|     AS|   490|   SEA| SAN|     135|    1050|  12|    36|      2001|Fixed wing multi ...|        BOEING|    737-990|      2|  149|   NA|Turbo-jet|
| N564AS|2014|   11| 19|    1812|       -3|    2352|       -4|     AS|    26|   SEA| ORD|     198|    1721|  18|    12|      2006|Fixed wing multi ...|        BOEING|    737-890|      2|  149|   NA|Turbo-fan|
| N323AS|2014|   11|  8|    1653|       -2|    1924|       -1|     AS|   448|   SEA| LAX|     130|     954|  16|    53|      2004|Fixed wing multi ...|        BOEING|    737-990|      2|  149|   NA|Turbo-jet|
| N305AS|2014|    8|  3|    1120|        0|    1415|        2|     AS|   656|   SEA| PHX|     154|    1107|  11|    20|      2001|Fixed wing multi ...|        BOEING|    737-990|      2|  149|   NA|Turbo-jet|
| N433AS|2014|   10| 30|     811|       21|    1038|       29|     AS|   608|   SEA| LAS|     127|     867|   8|    11|      2013|Fixed wing multi ...|        BOEING|  737-990ER|      2|  222|   NA|Turbo-fan|
| N765AS|2014|   11| 12|    2346|       -4|     217|      -28|     AS|   121|   SEA| ANC|     183|    1448|  23|    46|      1992|Fixed wing multi ...|        BOEING|    737-4Q8|      2|  149|   NA|Turbo-fan|
| N713AS|2014|   10| 31|    1314|       89|    1544|      111|     AS|   306|   SEA| SFO|     129|     679|  13|    14|      1999|Fixed wing multi ...|        BOEING|    737-490|      2|  149|   NA|Turbo-jet|
| N27205|2014|    1| 29|    2009|        3|    2159|        9|     UA|  1458|   PDX| SFO|      90|     550|  20|     9|      2000|Fixed wing multi ...|        BOEING|    737-824|      2|  149|   NA|Turbo-fan|
| N626AS|2014|   12| 17|    2015|       50|    2150|       41|     AS|   368|   SEA| SMF|      76|     605|  20|    15|      2001|Fixed wing multi ...|        BOEING|    737-790|      2|  151|   NA|Turbo-jet|
| N8634A|2014|    8| 11|    1017|       -3|    1613|       -7|     WN|   827|   SEA| MDW|     216|    1733|  10|    17|      2014|Fixed wing multi ...|        BOEING|    737-8H4|      2|  140|   NA|Turbo-fan|
| N597AS|2014|    1| 13|    2156|       -9|     607|      -15|     AS|    24|   SEA| BOS|     290|    2496|  21|    56|      2008|Fixed wing multi ...|        BOEING|    737-890|      2|  149|   NA|Turbo-fan|
| N215AG|2014|    6|  5|    1733|      -12|    1945|      -10|     OO|  3488|   PDX| BUR|     111|     817|  17|    33|      2001|Fixed wing multi ...|BOMBARDIER INC|CL-600-2C10|      2|   80|   NA|Turbo-fan|
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+--------------------+--------------+-----------+-------+-----+-----+---------+
only showing top 20 rows

None
'''

# Data types
'''
Good work! Before you get started modeling, it's important to know that Spark only handles numeric data. That means all of the columns in your DataFrame must be either integers or decimals (called 'doubles' in Spark).

When we imported our data, we let Spark guess what kind of information each column held. Unfortunately, Spark doesn't always guess right and you can see that some of the columns in our DataFrame are strings containing numbers as opposed to actual numeric values.

!!! To remedy this, you can use the .cast() method in combination with the .withColumn() method. It's important to note that .cast() works on columns, while .withColumn() works on DataFrames.

The only argument you need to pass to .cast() is the kind of value you want to create, in string form. For example, to create integers, you'll pass the argument "integer" and for decimal numbers you'll use "double".

You can put this call to .cast() inside a call to .withColumn() to overwrite the already existing column, just like you did in the previous chapter!

What kind of data does Spark need for modeling?

Doubles
Integers
Decimals
Numeric √  Integers + Decimals(Doubles)
Strings
'''

# String to integer
'''
Now you'll use the .cast() method you learned in the previous exercise to convert all the appropriate columns from your DataFrame model_data to integers!

To convert the type of a column using the .cast() method, you can write code like this:

dataframe = dataframe.withColumn("col", dataframe.col.cast("new_type"))
Instructions
Use the method .withColumn() to .cast() the following columns to type "integer". Access the columns using the df.col notation:
model_data.arr_delay
model_data.air_time
model_data.month
model_data.plane_year
'''

# Cast the columns to integers
model_data = model_data.withColumn("arr_delay", model_data.arr_delay.cast("integer"))
model_data = model_data.withColumn("air_time", model_data.air_time.cast("integer"))
model_data = model_data.withColumn("month", model_data.month.cast("integer"))
model_data = model_data.withColumn("plane_year", model_data.plane_year.cast("integer"))


# print(model_data.show())
# don't change the DataFrame structure, just change the data type, maybe it's because the column name is tha same


# Create a new column
'''
In the last exercise, you converted the column plane_year to an integer. This column holds the year each plane was manufactured. However, your model will use the planes' age, which is slightly different from the year it was made!

Instructions
Create the column plane_age using the .withColumn() method and subtracting the year of manufacture (column plane_year) from the year (column year) of the flight.
'''

# Create the column plane_age
model_data = model_data.withColumn("plane_age", model_data.year - model_data.plane_year)

# Making a Boolean
'''
Consider that you're modeling a yes or no question: is the flight late? However, your data contains the arrival delay in minutes for each flight. Thus, you'll need to create a boolean column which indicates whether the flight was late or not!

Instructions
Use the .withColumn() method to create the column is_late. This column is equal to model_data.arr_delay > 0.
Convert this column to an integer column so that you can use it in your model and name it label (this is the default name for the response variable in Spark's machine learning routines).
Filter out missing values (this has been done for you).
'''

# Create is_late
model_data = model_data.withColumn("is_late", model_data.arr_delay > 0)

# Convert to an integer
model_data = model_data.withColumn("label", model_data.is_late.cast("integer"))

# Remove missing values
model_data = model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")

'''
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+--------------------+--------------+-----------+-------+-----+-----+---------+---------+-------+-----+
|tailnum|year|month|day|dep_time|dep_delay|arr_time|arr_delay|carrier|flight|origin|dest|air_time|distance|hour|minute|plane_year|                type|  manufacturer|      model|engines|seats|speed|   engine|plane_age|is_late|label|
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+--------------------+--------------+-----------+-------+-----+-----+---------+---------+-------+-----+
| N846VA|2014|   12|  8|     658|       -7|     935|       -5|     VX|  1780|   SEA| LAX|     132|     954|   6|    58|      2011|Fixed wing multi ...|        AIRBUS|   A320-214|      2|  182|   NA|Turbo-fan|      3.0|  false|    0|
| N559AS|2014|    1| 22|    1040|        5|    1505|        5|     AS|   851|   SEA| HNL|     360|    2677|  10|    40|      2006|Fixed wing multi ...|        BOEING|    737-890|      2|  149|   NA|Turbo-fan|      8.0|   true|    1|
| N847VA|2014|    3|  9|    1443|       -2|    1652|        2|     VX|   755|   SEA| SFO|     111|     679|  14|    43|      2011|Fixed wing multi ...|        AIRBUS|   A320-214|      2|  182|   NA|Turbo-fan|      3.0|   true|    1|
| N360SW|2014|    4|  9|    1705|       45|    1839|       34|     WN|   344|   PDX| SJC|      83|     569|  17|     5|      1992|Fixed wing multi ...|        BOEING|    737-3H4|      2|  149|   NA|Turbo-fan|     22.0|   true|    1|
| N612AS|2014|    3|  9|     754|       -1|    1015|        1|     AS|   522|   SEA| BUR|     127|     937|   7|    54|      1999|Fixed wing multi ...|        BOEING|    737-790|      2|  151|   NA|Turbo-jet|     15.0|   true|    1|
| N646SW|2014|    1| 15|    1037|        7|    1352|        2|     WN|    48|   PDX| DEN|     121|     991|  10|    37|      1997|Fixed wing multi ...|        BOEING|    737-3H4|      2|  149|   NA|Turbo-fan|     17.0|   true|    1|
| N422WN|2014|    7|  2|     847|       42|    1041|       51|     WN|  1520|   PDX| OAK|      90|     543|   8|    47|      2002|Fixed wing multi ...|        BOEING|    737-7H4|      2|  140|   NA|Turbo-fan|     12.0|   true|    1|
| N361VA|2014|    5| 12|    1655|       -5|    1842|      -18|     VX|   755|   SEA| SFO|      98|     679|  16|    55|      2013|Fixed wing multi ...|        AIRBUS|   A320-214|      2|  182|   NA|Turbo-fan|      1.0|  false|    0|
| N309AS|2014|    4| 19|    1236|       -4|    1508|       -7|     AS|   490|   SEA| SAN|     135|    1050|  12|    36|      2001|Fixed wing multi ...|        BOEING|    737-990|      2|  149|   NA|Turbo-jet|     13.0|  false|    0|
| N564AS|2014|   11| 19|    1812|       -3|    2352|       -4|     AS|    26|   SEA| ORD|     198|    1721|  18|    12|      2006|Fixed wing multi ...|        BOEING|    737-890|      2|  149|   NA|Turbo-fan|      8.0|  false|    0|
| N323AS|2014|   11|  8|    1653|       -2|    1924|       -1|     AS|   448|   SEA| LAX|     130|     954|  16|    53|      2004|Fixed wing multi ...|        BOEING|    737-990|      2|  149|   NA|Turbo-jet|     10.0|  false|    0|
| N305AS|2014|    8|  3|    1120|        0|    1415|        2|     AS|   656|   SEA| PHX|     154|    1107|  11|    20|      2001|Fixed wing multi ...|        BOEING|    737-990|      2|  149|   NA|Turbo-jet|     13.0|   true|    1|
| N433AS|2014|   10| 30|     811|       21|    1038|       29|     AS|   608|   SEA| LAS|     127|     867|   8|    11|      2013|Fixed wing multi ...|        BOEING|  737-990ER|      2|  222|   NA|Turbo-fan|      1.0|   true|    1|
| N765AS|2014|   11| 12|    2346|       -4|     217|      -28|     AS|   121|   SEA| ANC|     183|    1448|  23|    46|      1992|Fixed wing multi ...|        BOEING|    737-4Q8|      2|  149|   NA|Turbo-fan|     22.0|  false|    0|
| N713AS|2014|   10| 31|    1314|       89|    1544|      111|     AS|   306|   SEA| SFO|     129|     679|  13|    14|      1999|Fixed wing multi ...|        BOEING|    737-490|      2|  149|   NA|Turbo-jet|     15.0|   true|    1|
| N27205|2014|    1| 29|    2009|        3|    2159|        9|     UA|  1458|   PDX| SFO|      90|     550|  20|     9|      2000|Fixed wing multi ...|        BOEING|    737-824|      2|  149|   NA|Turbo-fan|     14.0|   true|    1|
| N626AS|2014|   12| 17|    2015|       50|    2150|       41|     AS|   368|   SEA| SMF|      76|     605|  20|    15|      2001|Fixed wing multi ...|        BOEING|    737-790|      2|  151|   NA|Turbo-jet|     13.0|   true|    1|
| N8634A|2014|    8| 11|    1017|       -3|    1613|       -7|     WN|   827|   SEA| MDW|     216|    1733|  10|    17|      2014|Fixed wing multi ...|        BOEING|    737-8H4|      2|  140|   NA|Turbo-fan|      0.0|  false|    0|
| N597AS|2014|    1| 13|    2156|       -9|     607|      -15|     AS|    24|   SEA| BOS|     290|    2496|  21|    56|      2008|Fixed wing multi ...|        BOEING|    737-890|      2|  149|   NA|Turbo-fan|      6.0|  false|    0|
| N215AG|2014|    6|  5|    1733|      -12|    1945|      -10|     OO|  3488|   PDX| BUR|     111|     817|  17|    33|      2001|Fixed wing multi ...|BOMBARDIER INC|CL-600-2C10|      2|   80|   NA|Turbo-fan|     13.0|  false|    0|
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+--------------------+--------------+-----------+-------+-----+-----+---------+---------+-------+-----+
only showing top 20 rows

None
'''

# Strings and factors
# creating an Estimator and then a Transformer
'''
As you know, Spark requires numeric data for modeling. So far this hasn't been an issue; even boolean columns can easily be converted to integers without any trouble. But you'll also be using the airline and the plane's destination as features in your model. These are coded as strings and there isn't any obvious way to convert them to a numeric data type.

Fortunately, PySpark has functions for handling this built into the pyspark.ml.features submodule. You can create what are called 'one-hot vectors' to represent the carrier and the destination of each flight. A one-hot vector is a way of representing a categorical feature where every observation has a vector in which all elements are zero except for at most one element, which has a value of one (1).

Each element in the vector corresponds to a level of the feature, so it's possible to tell what the right level is by seeing which element of the vector is equal to one (1).

The first step to encoding your categorical feature is to create a StringIndexer. Members of this class are Estimators that take a DataFrame with a column of strings and map each unique string to a number. Then, the Estimator returns a Transformer that takes a DataFrame, attaches the mapping to it as metadata, and returns a new DataFrame with a numeric column corresponding to the string column.

The second step is to encode this numeric column as a one-hot vector using a OneHotEncoder. This works exactly the same way as the StringIndexer by creating an Estimator and then a Transformer. The end result is a column that encodes your categorical feature as a vector that's suitable for machine learning routines!

!! This may seem complicated, but don't worry! All you have to remember is that you need to create a StringIndexer and a OneHotEncoder, and the Pipeline will take care of the rest.

Why do you have to encode a categorical feature as a one-hot vector?

Answer the question

It makes fitting the model faster.
Spark can only model numeric features. √
For compatibility with scikit-learn.
'''


# Carrier
'''
In this exercise you'll create a StringIndexer and a OneHotEncoder to code the carrier column. To do this, you'll call the class constructors with the arguments inputCol and outputCol.

The inputCol is the name of the column you want to index or encode, and the outputCol is the name of the new column that the Transformer should create.

Instructions

Create a StringIndexer called carr_indexer by calling StringIndexer() with inputCol="carrier" and outputCol="carrier_index".
Create a OneHotEncoder called carr_encoder by calling OneHotEncoder() with inputCol="carrier_index" and outputCol="carrier_fact".
'''

# Create a StringIndexer
carr_indexer = StringIndexer(inputCol="carrier", outputCol="carrier_index")

# Create a OneHotEncoder
carr_encoder = OneHotEncoder(inputCol="carrier_index", outputCol="carrier_fact")


# Destination
'''
Now you'll encode the dest column just like you did in the previous exercise.

Instructions
Create a StringIndexer called dest_indexer by calling StringIndexer() with inputCol="dest" and outputCol="dest_index".
Create a OneHotEncoder called dest_encoder by calling OneHotEncoder() with inputCol="dest_index" and outputCol="dest_fact".
'''

# Create a StringIndexer
dest_indexer = StringIndexer(inputCol="dest", outputCol="dest_index")

# Create a OneHotEncoder
dest_encoder = OneHotEncoder(inputCol="dest_index", outputCol="dest_fact")


# Assemble a vector
'''
The last step in the Pipeline is to combine all of the columns containing our features into a single column. This has to be done before modeling can take place because every Spark modeling routine expects the data to be in this form. You can do this by storing each of the values from a column as an entry in a vector. Then, from the model's point of view, every observation is a vector that contains all of the information about it and a label that tells the modeler what value that observation corresponds to.

Because of this, the pyspark.ml.feature submodule contains a class called VectorAssembler. This Transformer takes all of the columns you specify and combines them into a new vector column.

Instructions
Create a VectorAssembler by calling VectorAssembler() with the inputCols names as a list and the outputCol name "features".
The list of columns should be ["month", "air_time", "carrier_fact", "dest_fact", "plane_age"].
'''

# Make a VectorAssembler
vec_assembler = VectorAssembler(inputCols=["month", "air_time", "carrier_fact", "dest_fact", "plane_age"], outputCol="features")

# print(vec_assembler)
# VectorAssembler_088f120309f1


# Create the pipeline
'''
You're finally ready to create a Pipeline!

Pipeline is a class in the pyspark.ml module that combines all the Estimators and Transformers that you've already created. This lets you reuse the same modeling process over and over again by wrapping it up in one simple object. Neat, right?

Instructions
Import Pipeline from pyspark.ml.
Call the Pipeline() constructor with the keyword argument stages to create a Pipeline called flights_pipe.
stages should be a list holding all the stages you want your data to go through in the pipeline. Here this is just: [dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler]
'''

# Import Pipeline
from pyspark.ml import Pipeline

# Make the pipeline
flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])


# Test vs. Train
'''
After you've cleaned your data and gotten it ready for modeling, one of the most important steps is to split the data into a test set and a train set. After that, don't touch your test data until you think you have a good model! As you're building models and forming hypotheses, you can test them on your training data to get an idea of their performance.

Once you've got your favorite model, you can see how well it predicts the new data in your test set. This never-before-seen data will give you a much more realistic idea of your model's performance in the real world when you're trying to predict or classify new data.

!! In Spark it's important to make sure you split the data after all the transformations. This is because operations like StringIndexer don't always produce the same index even when given the same list of strings.

Why is it important to use a test set in model evaluation?
Evaluating your model improves its accuracy.
By evaluating your model with a test set you can get a good idea of performance on new data. √
Using a test set lets you check your code for errors.
'''


# Transform the data
# apply pipeline to model_data
'''
Hooray, now you're finally ready to pass your data through the Pipeline you created!

Instructions
Create the DataFrame piped_data by calling the Pipeline methods .fit() and .transform() in a chain. Both of these methods take model_data as their only argument.
'''

# Fit and transform the data
piped_data = flights_pipe.fit(model_data).transform(model_data)

# print(model_data.show())
# print(piped_data.show())
'''
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+--------------------+--------------+-----------+-------+-----+-----+---------+---------+-------+-----+
|tailnum|year|month|day|dep_time|dep_delay|arr_time|arr_delay|carrier|flight|origin|dest|air_time|distance|hour|minute|plane_year|                type|  manufacturer|      model|engines|seats|speed|   engine|plane_age|is_late|label|
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+--------------------+--------------+-----------+-------+-----+-----+---------+---------+-------+-----+
| N846VA|2014|   12|  8|     658|       -7|     935|       -5|     VX|  1780|   SEA| LAX|     132|     954|   6|    58|      2011|Fixed wing multi ...|        AIRBUS|   A320-214|      2|  182|   NA|Turbo-fan|      3.0|  false|    0|
| N559AS|2014|    1| 22|    1040|        5|    1505|        5|     AS|   851|   SEA| HNL|     360|    2677|  10|    40|      2006|Fixed wing multi ...|        BOEING|    737-890|      2|  149|   NA|Turbo-fan|      8.0|   true|    1|
| N847VA|2014|    3|  9|    1443|       -2|    1652|        2|     VX|   755|   SEA| SFO|     111|     679|  14|    43|      2011|Fixed wing multi ...|        AIRBUS|   A320-214|      2|  182|   NA|Turbo-fan|      3.0|   true|    1|
| N360SW|2014|    4|  9|    1705|       45|    1839|       34|     WN|   344|   PDX| SJC|      83|     569|  17|     5|      1992|Fixed wing multi ...|        BOEING|    737-3H4|      2|  149|   NA|Turbo-fan|     22.0|   true|    1|
| N612AS|2014|    3|  9|     754|       -1|    1015|        1|     AS|   522|   SEA| BUR|     127|     937|   7|    54|      1999|Fixed wing multi ...|        BOEING|    737-790|      2|  151|   NA|Turbo-jet|     15.0|   true|    1|
| N646SW|2014|    1| 15|    1037|        7|    1352|        2|     WN|    48|   PDX| DEN|     121|     991|  10|    37|      1997|Fixed wing multi ...|        BOEING|    737-3H4|      2|  149|   NA|Turbo-fan|     17.0|   true|    1|
| N422WN|2014|    7|  2|     847|       42|    1041|       51|     WN|  1520|   PDX| OAK|      90|     543|   8|    47|      2002|Fixed wing multi ...|        BOEING|    737-7H4|      2|  140|   NA|Turbo-fan|     12.0|   true|    1|
| N361VA|2014|    5| 12|    1655|       -5|    1842|      -18|     VX|   755|   SEA| SFO|      98|     679|  16|    55|      2013|Fixed wing multi ...|        AIRBUS|   A320-214|      2|  182|   NA|Turbo-fan|      1.0|  false|    0|
| N309AS|2014|    4| 19|    1236|       -4|    1508|       -7|     AS|   490|   SEA| SAN|     135|    1050|  12|    36|      2001|Fixed wing multi ...|        BOEING|    737-990|      2|  149|   NA|Turbo-jet|     13.0|  false|    0|
| N564AS|2014|   11| 19|    1812|       -3|    2352|       -4|     AS|    26|   SEA| ORD|     198|    1721|  18|    12|      2006|Fixed wing multi ...|        BOEING|    737-890|      2|  149|   NA|Turbo-fan|      8.0|  false|    0|
| N323AS|2014|   11|  8|    1653|       -2|    1924|       -1|     AS|   448|   SEA| LAX|     130|     954|  16|    53|      2004|Fixed wing multi ...|        BOEING|    737-990|      2|  149|   NA|Turbo-jet|     10.0|  false|    0|
| N305AS|2014|    8|  3|    1120|        0|    1415|        2|     AS|   656|   SEA| PHX|     154|    1107|  11|    20|      2001|Fixed wing multi ...|        BOEING|    737-990|      2|  149|   NA|Turbo-jet|     13.0|   true|    1|
| N433AS|2014|   10| 30|     811|       21|    1038|       29|     AS|   608|   SEA| LAS|     127|     867|   8|    11|      2013|Fixed wing multi ...|        BOEING|  737-990ER|      2|  222|   NA|Turbo-fan|      1.0|   true|    1|
| N765AS|2014|   11| 12|    2346|       -4|     217|      -28|     AS|   121|   SEA| ANC|     183|    1448|  23|    46|      1992|Fixed wing multi ...|        BOEING|    737-4Q8|      2|  149|   NA|Turbo-fan|     22.0|  false|    0|
| N713AS|2014|   10| 31|    1314|       89|    1544|      111|     AS|   306|   SEA| SFO|     129|     679|  13|    14|      1999|Fixed wing multi ...|        BOEING|    737-490|      2|  149|   NA|Turbo-jet|     15.0|   true|    1|
| N27205|2014|    1| 29|    2009|        3|    2159|        9|     UA|  1458|   PDX| SFO|      90|     550|  20|     9|      2000|Fixed wing multi ...|        BOEING|    737-824|      2|  149|   NA|Turbo-fan|     14.0|   true|    1|
| N626AS|2014|   12| 17|    2015|       50|    2150|       41|     AS|   368|   SEA| SMF|      76|     605|  20|    15|      2001|Fixed wing multi ...|        BOEING|    737-790|      2|  151|   NA|Turbo-jet|     13.0|   true|    1|
| N8634A|2014|    8| 11|    1017|       -3|    1613|       -7|     WN|   827|   SEA| MDW|     216|    1733|  10|    17|      2014|Fixed wing multi ...|        BOEING|    737-8H4|      2|  140|   NA|Turbo-fan|      0.0|  false|    0|
| N597AS|2014|    1| 13|    2156|       -9|     607|      -15|     AS|    24|   SEA| BOS|     290|    2496|  21|    56|      2008|Fixed wing multi ...|        BOEING|    737-890|      2|  149|   NA|Turbo-fan|      6.0|  false|    0|
| N215AG|2014|    6|  5|    1733|      -12|    1945|      -10|     OO|  3488|   PDX| BUR|     111|     817|  17|    33|      2001|Fixed wing multi ...|BOMBARDIER INC|CL-600-2C10|      2|   80|   NA|Turbo-fan|     13.0|  false|    0|
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+--------------------+--------------+-----------+-------+-----+-----+---------+---------+-------+-----+
only showing top 20 rows

None

+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+--------------------+--------------+-----------+-------+-----+-----+---------+---------+-------+-----+----------+---------------+-------------+--------------+--------------------+
|tailnum|year|month|day|dep_time|dep_delay|arr_time|arr_delay|carrier|flight|origin|dest|air_time|distance|hour|minute|plane_year|                type|  manufacturer|      model|engines|seats|speed|   engine|plane_age|is_late|label|dest_index|      dest_fact|carrier_index|  carrier_fact|            features|
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+--------------------+--------------+-----------+-------+-----+-----+---------+---------+-------+-----+----------+---------------+-------------+--------------+--------------------+
| N846VA|2014|   12|  8|     658|       -7|     935|       -5|     VX|  1780|   SEA| LAX|     132|     954|   6|    58|      2011|Fixed wing multi ...|        AIRBUS|   A320-214|      2|  182|   NA|Turbo-fan|      3.0|  false|    0|       1.0| (68,[1],[1.0])|          7.0|(10,[7],[1.0])|(81,[0,1,3,77,80]...|
| N559AS|2014|    1| 22|    1040|        5|    1505|        5|     AS|   851|   SEA| HNL|     360|    2677|  10|    40|      2006|Fixed wing multi ...|        BOEING|    737-890|      2|  149|   NA|Turbo-fan|      8.0|   true|    1|      19.0|(68,[19],[1.0])|          0.0|(10,[0],[1.0])|(81,[0,1,21,70,80...|
| N847VA|2014|    3|  9|    1443|       -2|    1652|        2|     VX|   755|   SEA| SFO|     111|     679|  14|    43|      2011|Fixed wing multi ...|        AIRBUS|   A320-214|      2|  182|   NA|Turbo-fan|      3.0|   true|    1|       0.0| (68,[0],[1.0])|          7.0|(10,[7],[1.0])|(81,[0,1,2,77,80]...|
| N360SW|2014|    4|  9|    1705|       45|    1839|       34|     WN|   344|   PDX| SJC|      83|     569|  17|     5|      1992|Fixed wing multi ...|        BOEING|    737-3H4|      2|  149|   NA|Turbo-fan|     22.0|   true|    1|       7.0| (68,[7],[1.0])|          1.0|(10,[1],[1.0])|(81,[0,1,9,71,80]...|
| N612AS|2014|    3|  9|     754|       -1|    1015|        1|     AS|   522|   SEA| BUR|     127|     937|   7|    54|      1999|Fixed wing multi ...|        BOEING|    737-790|      2|  151|   NA|Turbo-jet|     15.0|   true|    1|      22.0|(68,[22],[1.0])|          0.0|(10,[0],[1.0])|(81,[0,1,24,70,80...|
| N646SW|2014|    1| 15|    1037|        7|    1352|        2|     WN|    48|   PDX| DEN|     121|     991|  10|    37|      1997|Fixed wing multi ...|        BOEING|    737-3H4|      2|  149|   NA|Turbo-fan|     17.0|   true|    1|       2.0| (68,[2],[1.0])|          1.0|(10,[1],[1.0])|(81,[0,1,4,71,80]...|
| N422WN|2014|    7|  2|     847|       42|    1041|       51|     WN|  1520|   PDX| OAK|      90|     543|   8|    47|      2002|Fixed wing multi ...|        BOEING|    737-7H4|      2|  140|   NA|Turbo-fan|     12.0|   true|    1|       8.0| (68,[8],[1.0])|          1.0|(10,[1],[1.0])|(81,[0,1,10,71,80...|
| N361VA|2014|    5| 12|    1655|       -5|    1842|      -18|     VX|   755|   SEA| SFO|      98|     679|  16|    55|      2013|Fixed wing multi ...|        AIRBUS|   A320-214|      2|  182|   NA|Turbo-fan|      1.0|  false|    0|       0.0| (68,[0],[1.0])|          7.0|(10,[7],[1.0])|(81,[0,1,2,77,80]...|
| N309AS|2014|    4| 19|    1236|       -4|    1508|       -7|     AS|   490|   SEA| SAN|     135|    1050|  12|    36|      2001|Fixed wing multi ...|        BOEING|    737-990|      2|  149|   NA|Turbo-jet|     13.0|  false|    0|      10.0|(68,[10],[1.0])|          0.0|(10,[0],[1.0])|(81,[0,1,12,70,80...|
| N564AS|2014|   11| 19|    1812|       -3|    2352|       -4|     AS|    26|   SEA| ORD|     198|    1721|  18|    12|      2006|Fixed wing multi ...|        BOEING|    737-890|      2|  149|   NA|Turbo-fan|      8.0|  false|    0|      11.0|(68,[11],[1.0])|          0.0|(10,[0],[1.0])|(81,[0,1,13,70,80...|
| N323AS|2014|   11|  8|    1653|       -2|    1924|       -1|     AS|   448|   SEA| LAX|     130|     954|  16|    53|      2004|Fixed wing multi ...|        BOEING|    737-990|      2|  149|   NA|Turbo-jet|     10.0|  false|    0|       1.0| (68,[1],[1.0])|          0.0|(10,[0],[1.0])|(81,[0,1,3,70,80]...|
| N305AS|2014|    8|  3|    1120|        0|    1415|        2|     AS|   656|   SEA| PHX|     154|    1107|  11|    20|      2001|Fixed wing multi ...|        BOEING|    737-990|      2|  149|   NA|Turbo-jet|     13.0|   true|    1|       4.0| (68,[4],[1.0])|          0.0|(10,[0],[1.0])|(81,[0,1,6,70,80]...|
| N433AS|2014|   10| 30|     811|       21|    1038|       29|     AS|   608|   SEA| LAS|     127|     867|   8|    11|      2013|Fixed wing multi ...|        BOEING|  737-990ER|      2|  222|   NA|Turbo-fan|      1.0|   true|    1|       3.0| (68,[3],[1.0])|          0.0|(10,[0],[1.0])|(81,[0,1,5,70,80]...|
| N765AS|2014|   11| 12|    2346|       -4|     217|      -28|     AS|   121|   SEA| ANC|     183|    1448|  23|    46|      1992|Fixed wing multi ...|        BOEING|    737-4Q8|      2|  149|   NA|Turbo-fan|     22.0|  false|    0|       5.0| (68,[5],[1.0])|          0.0|(10,[0],[1.0])|(81,[0,1,7,70,80]...|
| N713AS|2014|   10| 31|    1314|       89|    1544|      111|     AS|   306|   SEA| SFO|     129|     679|  13|    14|      1999|Fixed wing multi ...|        BOEING|    737-490|      2|  149|   NA|Turbo-jet|     15.0|   true|    1|       0.0| (68,[0],[1.0])|          0.0|(10,[0],[1.0])|(81,[0,1,2,70,80]...|
| N27205|2014|    1| 29|    2009|        3|    2159|        9|     UA|  1458|   PDX| SFO|      90|     550|  20|     9|      2000|Fixed wing multi ...|        BOEING|    737-824|      2|  149|   NA|Turbo-fan|     14.0|   true|    1|       0.0| (68,[0],[1.0])|          4.0|(10,[4],[1.0])|(81,[0,1,2,74,80]...|
| N626AS|2014|   12| 17|    2015|       50|    2150|       41|     AS|   368|   SEA| SMF|      76|     605|  20|    15|      2001|Fixed wing multi ...|        BOEING|    737-790|      2|  151|   NA|Turbo-jet|     13.0|   true|    1|       9.0| (68,[9],[1.0])|          0.0|(10,[0],[1.0])|(81,[0,1,11,70,80...|
| N8634A|2014|    8| 11|    1017|       -3|    1613|       -7|     WN|   827|   SEA| MDW|     216|    1733|  10|    17|      2014|Fixed wing multi ...|        BOEING|    737-8H4|      2|  140|   NA|Turbo-fan|      0.0|  false|    0|      31.0|(68,[31],[1.0])|          1.0|(10,[1],[1.0])|(81,[0,1,33,71],[...|
| N597AS|2014|    1| 13|    2156|       -9|     607|      -15|     AS|    24|   SEA| BOS|     290|    2496|  21|    56|      2008|Fixed wing multi ...|        BOEING|    737-890|      2|  149|   NA|Turbo-fan|      6.0|  false|    0|      24.0|(68,[24],[1.0])|          0.0|(10,[0],[1.0])|(81,[0,1,26,70,80...|
| N215AG|2014|    6|  5|    1733|      -12|    1945|      -10|     OO|  3488|   PDX| BUR|     111|     817|  17|    33|      2001|Fixed wing multi ...|BOMBARDIER INC|CL-600-2C10|      2|   80|   NA|Turbo-fan|     13.0|  false|    0|      22.0|(68,[22],[1.0])|          2.0|(10,[2],[1.0])|(81,[0,1,24,72,80...|
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+--------------------+--------------+-----------+-------+-----+-----+---------+---------+-------+-----+----------+---------------+-------------+--------------+--------------------+
only showing top 20 rows

None
'''


# Split the data
'''
Now that you've done all your manipulations, the last step before modeling is to split the data!

Instructions
Use the DataFrame method .randomSplit() to split piped_data into two pieces, training with 60% of the data, and test with 40% of the data by passing the list [.6, .4] to the .randomSplit() method.
'''

# Split the data into training and test sets
training, test = piped_data.randomSplit([.6, .4])

# print(training.show())
# print(test.show())

'''
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+---------+-------+-----+----------+---------------+-------------+--------------+--------------------+
|tailnum|year|month|day|dep_time|dep_delay|arr_time|arr_delay|carrier|flight|origin|dest|air_time|distance|hour|minute|plane_year|plane_age|is_late|label|dest_index|      dest_fact|carrier_index|  carrier_fact|            features|
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+---------+-------+-----+----------+---------------+-------------+--------------+--------------------+
| N102UW|2014|    1| 12|     831|       -4|    1618|      -21|     US|  1883|   SEA| PHL|     265|    2378|   8|    31|      1998|     16.0|  false|    0|      30.0|(70,[30],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,32,77,82...|
| N102UW|2014|    2| 25|    1315|       -5|    2103|       -7|     US|  1805|   SEA| CLT|     256|    2279|  13|    15|      1998|     16.0|  false|    0|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N102UW|2014|    5|  7|    1311|        6|    2115|        2|     US|  1971|   SEA| CLT|     274|    2279|  13|    11|      1998|     16.0|   true|    1|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N102UW|2014|    5|  9|    1057|       -3|    1910|        2|     US|  2092|   SEA| CLT|     289|    2279|  10|    57|      1998|     16.0|   true|    1|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N102UW|2014|    6| 14|    2209|       -6|     557|       -4|     US|  1930|   PDX| CLT|     268|    2282|  22|     9|      1998|     16.0|  false|    0|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N102UW|2014|    9|  7|     830|       -1|    1637|        2|     US|   669|   PDX| PHL|     288|    2406|   8|    30|      1998|     16.0|   true|    1|      30.0|(70,[30],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,32,77,82...|
| N102UW|2014|   10| 10|    2212|       -3|     538|      -22|     US|  1930|   PDX| CLT|     247|    2282|  22|    12|      1998|     16.0|  false|    0|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N103US|2014|    2| 26|    1330|       10|    2112|        2|     US|  1805|   SEA| CLT|     255|    2279|  13|    30|      1999|     15.0|   true|    1|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N103US|2014|    3| 28|    2219|       -6|     559|       -5|     US|  1930|   PDX| CLT|     254|    2282|  22|    19|      1999|     15.0|  false|    0|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N103US|2014|    4| 10|    2213|       -2|     547|      -23|     US|  1930|   PDX| CLT|     254|    2282|  22|    13|      1999|     15.0|  false|    0|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N103US|2014|    5| 20|    1059|       -1|    1837|      -31|     US|  2092|   SEA| CLT|     258|    2279|  10|    59|      1999|     15.0|  false|    0|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N103US|2014|    6| 13|    1107|        7|    1912|       14|     US|  2092|   SEA| CLT|     270|    2279|  11|     7|      1999|     15.0|   true|    1|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N103US|2014|    6|  5|    1054|       -6|    1851|       -7|     US|  2092|   SEA| CLT|     264|    2279|  10|    54|      1999|     15.0|  false|    0|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N103US|2014|    6|  9|    1055|       -5|    1838|      -20|     US|  2092|   SEA| CLT|     259|    2279|  10|    55|      1999|     15.0|  false|    0|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N103US|2014|    7| 25|      38|       -7|     825|      -21|     US|  1917|   SEA| PHL|     273|    2378|   0|    38|      1999|     15.0|  false|    0|      30.0|(70,[30],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,32,77,82...|
| N103US|2014|    8| 30|      41|       -4|     854|        4|     US|   824|   SEA| PHL|     292|    2378|   0|    41|      1999|     15.0|   true|    1|      30.0|(70,[30],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,32,77,82...|
| N103US|2014|    9| 13|    1113|        3|    1852|      -26|     US|   728|   SEA| PHL|     259|    2378|  11|    13|      1999|     15.0|  false|    0|      30.0|(70,[30],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,32,77,82...|
| N103US|2014|    9| 21|     727|       12|    1519|       15|     US|  2046|   PDX| CLT|     267|    2282|   7|    27|      1999|     15.0|   true|    1|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N103US|2014|    9|  8|     610|       -5|    1424|        6|     US|   616|   SEA| PHL|     278|    2378|   6|    10|      1999|     15.0|   true|    1|      30.0|(70,[30],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,32,77,82...|
| N103US|2014|   11| 21|    2224|       -1|     555|      -11|     US|  1930|   PDX| CLT|     251|    2282|  22|    24|      1999|     15.0|  false|    0|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+---------+-------+-----+----------+---------------+-------------+--------------+--------------------+
only showing top 20 rows

None
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+---------+-------+-----+----------+---------------+-------------+--------------+--------------------+
|tailnum|year|month|day|dep_time|dep_delay|arr_time|arr_delay|carrier|flight|origin|dest|air_time|distance|hour|minute|plane_year|plane_age|is_late|label|dest_index|      dest_fact|carrier_index|  carrier_fact|            features|
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+---------+-------+-----+----------+---------------+-------------+--------------+--------------------+
| N102UW|2014|    2|  3|     832|       -3|    1709|       30|     US|  1935|   SEA| PHL|     274|    2378|   8|    32|      1998|     16.0|   true|    1|      30.0|(70,[30],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,32,77,82...|
| N102UW|2014|    4|  5|    2213|       -2|     548|      -22|     US|  1930|   PDX| CLT|     255|    2282|  22|    13|      1998|     16.0|  false|    0|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N102UW|2014|    5| 10|    2225|       10|     603|       -7|     US|  1930|   PDX| CLT|     262|    2282|  22|    25|      1998|     16.0|  false|    0|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N102UW|2014|    5| 15|    1322|       17|    2113|        0|     US|  1971|   SEA| CLT|     268|    2279|  13|    22|      1998|     16.0|  false|    0|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N102UW|2014|    5| 16|    1256|       -9|    2026|      -47|     US|  1971|   SEA| CLT|     247|    2279|  12|    56|      1998|     16.0|  false|    0|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N102UW|2014|    5| 18|    1257|       -8|    2047|      -26|     US|  1971|   SEA| CLT|     260|    2279|  12|    57|      1998|     16.0|  false|    0|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N102UW|2014|    5| 20|    1329|       24|    2058|      -15|     US|  1971|   SEA| CLT|     251|    2279|  13|    29|      1998|     16.0|  false|    0|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N102UW|2014|    7| 26|      38|       -7|     811|      -35|     US|  2051|   SEA| PHL|     258|    2378|   0|    38|      1998|     16.0|  false|    0|      30.0|(70,[30],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,32,77,82...|
| N102UW|2014|    7| 27|      52|        7|     824|      -22|     US|  2051|   SEA| PHL|     252|    2378|   0|    52|      1998|     16.0|  false|    0|      30.0|(70,[30],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,32,77,82...|
| N102UW|2014|    8| 18|     815|       -5|    1630|        2|     US|   798|   SEA| PHL|     295|    2378|   8|    15|      1998|     16.0|   true|    1|      30.0|(70,[30],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,32,77,82...|
| N102UW|2014|    8|  6|      40|       -5|     831|      -15|     US|  1917|   SEA| PHL|     268|    2378|   0|    40|      1998|     16.0|  false|    0|      30.0|(70,[30],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,32,77,82...|
| N102UW|2014|   11|  9|    2220|       -5|     555|      -11|     US|  1930|   PDX| CLT|     257|    2282|  22|    20|      1998|     16.0|  false|    0|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N103US|2014|    1| 28|     829|       -6|    1628|      -11|     US|  1935|   SEA| PHL|     272|    2378|   8|    29|      1999|     15.0|  false|    0|      30.0|(70,[30],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,32,77,82...|
| N103US|2014|    3| 14|    2218|       -7|     546|      -18|     US|  1930|   PDX| CLT|     251|    2282|  22|    18|      1999|     15.0|  false|    0|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N103US|2014|    4|  3|    1330|       10|    2130|       20|     US|  1805|   SEA| CLT|     275|    2279|  13|    30|      1999|     15.0|   true|    1|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N103US|2014|   12| 27|    2243|       13|     612|        1|     US|  1930|   PDX| CLT|     252|    2282|  22|    43|      1999|     15.0|   true|    1|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
| N104UW|2014|    1| 11|     843|        8|    1700|       21|     US|   788|   SEA| PHL|     282|    2378|   8|    43|      1999|     15.0|   true|    1|      30.0|(70,[30],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,32,77,82...|
| N104UW|2014|    1| 22|     836|        1|    1629|      -10|     US|  1935|   SEA| PHL|     267|    2378|   8|    36|      1999|     15.0|  false|    0|      30.0|(70,[30],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,32,77,82...|
| N104UW|2014|    9| 18|     614|       -1|    1403|      -15|     US|   616|   SEA| PHL|     269|    2378|   6|    14|      1999|     15.0|  false|    0|      30.0|(70,[30],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,32,77,82...|
| N104UW|2014|   11| 28|    2223|       -2|     541|      -25|     US|  1930|   PDX| CLT|     239|    2282|  22|    23|      1999|     15.0|  false|    0|      34.0|(70,[34],[1.0])|          5.0|(10,[5],[1.0])|(83,[0,1,36,77,82...|
+-------+----+-----+---+--------+---------+--------+---------+-------+------+------+----+--------+--------+----+------+----------+---------+-------+-----+----------+---------------+-------------+--------------+--------------------+
only showing top 20 rows

None
'''


