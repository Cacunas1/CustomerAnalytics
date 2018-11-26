
# Set-up environment ------------------------------------------------------

rm(list = ls())

library(caret)
library(caTools)
library(dplyr)
library(e1071)
library(ggplot2)
library(magrittr)
library(randomForest)
library(ranger)
library(RColorBrewer)
library(readr)
library(rpart)
library(SnowballC)
library(tictoc)
library(tm)
library(wordcloud)
library(zoo)
#library(kknn)

#Importing the dataset

orange_train <- read_tsv("./data/orange_small_train.data")
orange_test  <- read_tsv("./data/orange_small_test.data")
objective    <- read_csv("./data/KDD_CUP_2009_OBJECTIVE_COLUMNS_LABELS.csv")

# asdf --------------------------------------------------------------------

# Encoding the target feature as factor (In this case, Churn)
objective$Churn[objective$Churn == -1] <- 0
Churn <- factor(objective$Churn, levels = c(0, 1))

#=========================================================================#
# JC: Logistic, Naive Bayes, Decision Tree, Random Forest.
# Cristian: K-NN, SVM, Kernel SVM
#=========================================================================#

#It is required to omit the columns which have 0 variance
NoVar <- function(dat) {
  out <- lapply(dat, function(x) length(unique(x)))
  want <- which(!out > 1)
  unlist(want)
}

train_svm <- subset(orange_train, select = -NoVar(orange_train))

dataset_svm <- cbind(train_svm[1:160], Churn)

#Replacing missing values with the respective mean of each colum
df <- dataset_svm
df[] <- lapply(df, na.aggregate)

# Splitting the dataset into the Training set and Test set
set.seed(321)
split_svm = sample.split(df$Churn, SplitRatio = 0.75)
training_set_svm = subset(df, split_svm == TRUE)
test_set_svm = subset(df, split == FALSE)

tic("svmLinear training:")

classifier_svm <- train(
	x = dataset_svm[, 1:160],
	y = dataset_svm$Churn,
 	method = svmLinear,
	metric = ifelse(is.factor(y), "Accuracy", "RMSE"),   
    maximize = ifelse(metric == "RMSE", FALSE, TRUE),)

toc()

# Predicting the Test set results
prob_pred_svm = predict(classifier_svm, type = 'response', newdata = test_set_svm[-161])
y_pred_svm = ifelse(prob_pred_svm > 0.5, 1, 0)


# Making the Confusion Matrix
cm_svm = table(test_set[, 161], y_pred_svm > 0.5)
# Results:
# True Negatives: 11573/11582 (99.92%)
# True Positives: 1/918 (0.11%)


# Performing the splitting to the rough dataset-------------------------------------------------------------

# Splitting the dataset into the Training set and Test set
train <- cbind(orange_train, Churn)

set.seed(123)
split = sample.split(train$Churn, SplitRatio = 0.75)
training_set = subset(train, split == TRUE)
test_set = subset(train, split == FALSE)

# Naive Bayes -------------------------------------------------------------

# Fitting NB to the Training set
#install.packages('e1071')
classifier_nb = naiveBayes(x = training_set[-231],
                           y = training_set$Churn)

# Predicting the Test set results
y_pred_nb = predict(classifier_nb, newdata = test_set[-231])

# Making the Confusion Matrix
cm_nb = table(test_set[, 231], y_pred_nb)
#Results:
# True Negatives: 2304/11582 (19.89%)
# True Positives: 822/918 (89.54%)


# Decision Tree -----------------------------------------------------------

# Fitting Decision Tree Classification to the Training set
#install.packages('rpart')

classifier_dt = rpart(formula = Churn ~ .,
                      data = training_set)

# Predicting the Test set results
y_pred_dt = predict(classifier_dt, newdata = test_set[-231], type = 'class')

# Making the Confusion Matrix
cm_dt = table(test_set[, 231], y_pred_dt)
#Results:
# True Negatives: 11164/11582 (96.39%)
# True Positives: 51/918 (5.56%)


# Random Forest -----------------------------------------------------------

#RF does not allow NA's in predictors
training_set[!complete.cases(training_set),]

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')

set.seed(123)
classifier_rf = randomForest(x = training_set[-231],
                             y = training_set$Churn,
                             ntree = 500)

# Predicting the Test set results
y_pred_rf = predict(classifier_rf, newdata = test_set[-231])

# Making the Confusion Matrix
cm_rf = table(test_set[, 231], y_pred_rf)
