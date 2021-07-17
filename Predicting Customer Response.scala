// Databricks notebook source
// MAGIC %md
// MAGIC 
// MAGIC # Prediction of Customer Subscription using Bank Direct Telemarketing Data 
// MAGIC 
// MAGIC The Project is related with direct marketing campaigns of a banking institutions. The campaigns were based on phone calls. Often, more than one contact to the same client was required, to know if the product (bank term deposit) would be subscribed or not.
// MAGIC 
// MAGIC 
// MAGIC To predict customer response  to  bank  direct marketing  by  applying  two classifiers  namely Decision  Tree and Logistic  Regression Model. 

// COMMAND ----------

// MAGIC %md ### Load Source Data

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val bankDF = sqlContext.read.format("csv")
// MAGIC   .option("header", "true")
// MAGIC   .option("inferSchema", "true")
// MAGIC   .option("delimiter", ",")
// MAGIC   .load("/FileStore/tables/ALY6110DATA.csv")
// MAGIC 
// MAGIC display(bankDF)

// COMMAND ----------

// DBTITLE 1,Print Schema
// MAGIC %scala
// MAGIC 
// MAGIC bankDF.printSchema();

// COMMAND ----------

// DBTITLE 1,Creating Temporary View from Dataframe. We do this to run SQL queries on data.
// MAGIC %scala
// MAGIC 
// MAGIC bankDF.createOrReplaceTempView("BankData_Temp")

// COMMAND ----------

// DBTITLE 1,Querying the Temporary View
// MAGIC %sql
// MAGIC 
// MAGIC select * from BankData_Temp;

// COMMAND ----------

// MAGIC %md
// MAGIC # Data Details:
// MAGIC 
// MAGIC Input variables:
// MAGIC 
// MAGIC ### Bank client data:
// MAGIC 1 - age (numeric)
// MAGIC 
// MAGIC 2 - job : type of job (categorical: 'admin.','blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'selfemployed', 'services', 'student', 'technician', 'unemployed','unknown')
// MAGIC 
// MAGIC 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
// MAGIC 
// MAGIC 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
// MAGIC 
// MAGIC 5 - default: has credit in default? (categorical: 'no','yes','unknown')
// MAGIC 
// MAGIC 6 - housing: has housing loan? (categorical: 'no','yes','unknown')
// MAGIC 
// MAGIC 7 - loan: has personal loan? (categorical: 'no','yes','unknown')
// MAGIC 
// MAGIC 8 - contact: contact communication type (categorical: 'cellular','telephone')
// MAGIC 
// MAGIC 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
// MAGIC 
// MAGIC 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
// MAGIC 
// MAGIC 11 - duration: last contact duration, in seconds (numeric).
// MAGIC 
// MAGIC 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
// MAGIC 
// MAGIC 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
// MAGIC 
// MAGIC 14 - previous: number of contacts performed before this campaign and for this client (numeric)
// MAGIC 
// MAGIC 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
// MAGIC 
// MAGIC 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
// MAGIC 
// MAGIC 17 - cons.price.idx: consumer price index - monthly indicator (numeric)
// MAGIC 
// MAGIC 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
// MAGIC 
// MAGIC 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
// MAGIC 
// MAGIC 20 - nr.employed: number of employees - quarterly indicator (numeric)
// MAGIC 
// MAGIC 21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #Exploratory Data Analysis (EDA)

// COMMAND ----------

// DBTITLE 1,Distribution of our Labels:
// MAGIC %md
// MAGIC 
// MAGIC Knowing distribution of target variable labels

// COMMAND ----------

// DBTITLE 1,Displaying the Number(count) of Subscribers to Product (Yes/No)
// MAGIC %sql
// MAGIC 
// MAGIC select y as Subcribed, count(y) as Subscribed_Count from BankData_Temp group by y;

// COMMAND ----------

// DBTITLE 1,Displaying Percentage Number(count) of Subscribers of Product 
// MAGIC %sql
// MAGIC 
// MAGIC select y as Subcribed, count(y) as Subscribed_Count from BankData_Temp group by y;

// COMMAND ----------

// DBTITLE 1,Displaying Age Distribution  
// MAGIC %sql
// MAGIC 
// MAGIC select age, count(age) from BankData_Temp group by age order by age;

// COMMAND ----------

// DBTITLE 1,Jobs
// MAGIC %sql
// MAGIC 
// MAGIC select job, count(job) from BankData_Temp group by job;

// COMMAND ----------

// DBTITLE 1,Marital Status
// MAGIC %sql
// MAGIC 
// MAGIC select marital, count(marital) from BankData_Temp group by marital;

// COMMAND ----------

// DBTITLE 1,Education Background
// MAGIC %sql
// MAGIC 
// MAGIC select education, count(education) from BankData_Temp group by education;

// COMMAND ----------

// DBTITLE 1,Having House Loan?
// MAGIC %sql 
// MAGIC 
// MAGIC select housing, count(housing) from BankData_Temp group by housing;

// COMMAND ----------

// DBTITLE 1,Having Prersonal Loan?
// MAGIC %sql
// MAGIC 
// MAGIC select loan, count(loan) from BankData_Temp group by loan;

// COMMAND ----------

// MAGIC %md 
// MAGIC ## Creating a Regression Model

// COMMAND ----------

// DBTITLE 1,Importing necessary libraries 
// MAGIC %scala
// MAGIC 
// MAGIC import org.apache.spark.sql.functions._
// MAGIC import org.apache.spark.sql.Row
// MAGIC import org.apache.spark.sql.types._
// MAGIC 
// MAGIC import org.apache.spark.ml.classification.LogisticRegression
// MAGIC import org.apache.spark.ml.feature.VectorAssembler
// MAGIC import org.apache.spark.ml.feature.StringIndexer

// COMMAND ----------

// MAGIC %md ### Prepare the Training Data
// MAGIC To train the regression model, we need a training data set that includes a vector of all features, and a label column. 

// COMMAND ----------

// MAGIC %md ###VectorAssembler()
// MAGIC 
// MAGIC It is a transformer that combines a given list of columns or variables into a single vector column. It is useful for combining raw features into a single feature vector, to train our ML models.

// COMMAND ----------

// DBTITLE 1,List all String Data Type Columns in an Array for further processing
// MAGIC %scala
// MAGIC 
// MAGIC var CharCols = Array("job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome", "y")

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ###StringIndexer
// MAGIC 
// MAGIC StringIndexer encodes a string column of labels to a column of label indices. Basically it converts all character variables with words into numeric format for the ML model to understand

// COMMAND ----------

// MAGIC %md ### Define the Pipeline
// MAGIC A predictive model often requires multiple stages of feature preparation. 
// MAGIC 
// MAGIC In this case, pipeline stages will be:
// MAGIC 
// MAGIC - A **StringIndexer** estimator that converts string values to indexes for categorical features
// MAGIC - A **VectorAssembler** that combines categorical features into a single vector
// MAGIC 
// MAGIC This will prepare the data to be used to train the models

// COMMAND ----------

// DBTITLE 1,Pipeline Execution
// MAGIC %scala
// MAGIC 
// MAGIC import org.apache.spark.ml.attribute.Attribute
// MAGIC import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
// MAGIC import org.apache.spark.ml.{Pipeline, PipelineModel}
// MAGIC 
// MAGIC val indexers = CharCols.map { colName =>
// MAGIC   new StringIndexer().setInputCol(colName).setOutputCol(colName + "_indexed")
// MAGIC }
// MAGIC 
// MAGIC val pipeline = new Pipeline()
// MAGIC                     .setStages(indexers)      
// MAGIC 
// MAGIC val BankData_Indexed = pipeline.fit(bankDF).transform(bankDF)

// COMMAND ----------

// DBTITLE 1,Here we can see that all categorical features are converted into numeric format indicating "double" datatype
// MAGIC %scala
// MAGIC 
// MAGIC BankData_Indexed.printSchema()

// COMMAND ----------

// MAGIC %md ### Split the Data
// MAGIC It is common practice when building supervised machine learning models to split the source data, using some of it to train the model and reserving some to test the trained model. In this project, we will use 70% of the data for training, and reserve 30% for testing. 

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val splits = BankData_Indexed.randomSplit(Array(0.7, 0.3))
// MAGIC val train = splits(0)
// MAGIC val test = splits(1)
// MAGIC // val train_rows = train.count()
// MAGIC // val test_rows = test.count()
// MAGIC // println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

// COMMAND ----------

// DBTITLE 1,VectorAssembler() that combines categorical features into a single vector
// MAGIC %scala
// MAGIC 
// MAGIC val assembler = new VectorAssembler().setInputCols(Array("age", "duration", "campaign", "pdays", "previous", "empvarrate", "conspriceidx", "consconfidx", "euribor3m", 
// MAGIC "nremployed", "job_indexed", "marital_indexed", "education_indexed", "default_indexed", "housing_indexed", "loan_indexed", "contact_indexed", "month_indexed", "day_of_week_indexed", "poutcome_indexed")).setOutputCol("variables")
// MAGIC 
// MAGIC val training= assembler.transform(train).select($"variables", $"y_indexed".alias("label"))
// MAGIC 
// MAGIC assembler

// COMMAND ----------

// MAGIC %md ### Train a Regression Model

// COMMAND ----------

// MAGIC %scala
// MAGIC import org.apache.spark.ml.classification.LogisticRegression
// MAGIC 
// MAGIC val RegModel = new LogisticRegression().setLabelCol("label").setFeaturesCol("variables").setMaxIter(10).setRegParam(0.3)
// MAGIC val model = RegModel.fit(training)

// COMMAND ----------

// MAGIC %md ### Prepare the Testing Data

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val testing = assembler.transform(test).select($"variables", $"y_indexed".alias("trueLabel"))
// MAGIC testing.show()

// COMMAND ----------

// MAGIC %md ### Test the Model
// MAGIC  

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val predictionReg = model.transform(testing)
// MAGIC val predictedReg = predictionReg.select("variables", "prediction", "trueLabel")
// MAGIC predictedReg.show()

// COMMAND ----------

// MAGIC %md ### Regression Model Evalation

// COMMAND ----------

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val evaluatorReg = new BinaryClassificationEvaluator().setLabelCol("trueLabel").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
val aucReg = evaluatorReg.evaluate(predictionReg)
println("AUC Value / Accuracy = " + (aucReg))

// COMMAND ----------

// MAGIC %md ### Through Logistic Regression Model we are getting 87.09% accuracy.

// COMMAND ----------

// MAGIC %md ### Training Decision tree 

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC import org.apache.spark.ml.classification.DecisionTreeClassificationModel
// MAGIC import org.apache.spark.ml.classification.DecisionTreeClassifier
// MAGIC import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// MAGIC 
// MAGIC val DecTree = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("variables")
// MAGIC 
// MAGIC val model = DecTree.fit(training)

// COMMAND ----------

// MAGIC %md ### Test the Decision Tree Model

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val predictionDecTree = model.transform(testing)
// MAGIC val predictedDecTree = predictionDecTree.select("variables", "prediction", "trueLabel")
// MAGIC predictedDecTree.show(100)

// COMMAND ----------

// MAGIC %md ### Decision Tree model Evalation

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val evaluatorDecTree = new MulticlassClassificationEvaluator()
// MAGIC   .setLabelCol("trueLabel")
// MAGIC   .setPredictionCol("prediction")
// MAGIC   .setMetricName("accuracy")
// MAGIC val accuracyDecTree = evaluatorDecTree.evaluate(predictionDecTree)

// COMMAND ----------

// MAGIC %md ### Accuracy of Decision Tree Classifier is 91.55%
