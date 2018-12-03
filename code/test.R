
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
library(tidyr)
library(tm)
library(wordcloud)
library(zoo)
#library(kknn)

# Importing the dataset --------------------------------------------------

orange_train <- read_tsv("./data/orange_small_train.data")
orange_test  <- read_tsv("./data/orange_small_test.data")
objective    <- read_csv("./data/KDD_CUP_2009_OBJECTIVE_COLUMNS_LABELS.csv")

# functions --------------------------------------------------------------

not_all_na <- function(x) {!all(is.na(x))}

char2factorNumber <- function(x) {
  if(x %>% is.character()) {
    x %<>% as.factor()
    x %<>% unclass()
  }
  return(x)
}

getmode <- function(v) {
  uniqv <- unique(v)
  return(uniqv[which.max(tabulate(match(v, uniqv)))])
}

NoVar <- function(dat) {
  dist <- lapply(dat, function(x) length(unique(x)))
  want <- unlist(dist) > 1
  return(want)
}

# character to number -----------------------------------------------------

# Discard fully null columns
orange_train %<>% select_if(not_all_na)
orange_test %<>% select_if(not_all_na)

# Save indexes of categorical/numerical variables
nums <- lapply(orange_train, is.numeric) %>% unlist()
cats <- lapply(orange_train, is.character) %>% unlist()

# Transform text columns into factor
orange_train %<>% lapply(char2factorNumber) %>% as_tibble()

orange_train_nums <- orange_train[nums]
orange_train_cats <- orange_train[cats]

for (i in 1:ncol(orange_train_nums)) {
  orange_train_nums[is.na(orange_train[i]), i] <-
    mean(orange_train_nums[i], na.rm = TRUE)
}

# replace numerical nulls for column mean, and categorical nulls by column mode
# for(i in 1:ncol(orange_train)){
#   if (nums[i]) {
#     #orange_train[is.na(orange_train[,i]), i] <- mean(orange_train[,i], na.rm = TRUE)
#     #replace_na(orange_train[i], mean(orange_train[i], na.rm = TRUE))
#   } else {
#     orange_train[is.na(orange_train[,i]), i] <- 0
#   }
# }

# Encoding the target feature as factor (In this case, Churn)
objective$Churn[objective$Churn == -1] <- 0
Churn <- factor(objective$Churn, levels = c(0, 1))

#=========================================================================#
# JC: Random Forest.
# Cristian: SVM
#=========================================================================#

# ETL ---------------------------------------------------------------------

#It is required to omit the columns which have 0 variance
train_svm <- subset(orange_train_nums, select = NoVar(orange_train_nums))

dataset_svm <- cbind(train_svm, Churn)

# Splitting the dataset into the Training set and Test set
set.seed(321)
split_svm         <- sample.split(dataset_svm$Churn, SplitRatio = 0.75)
training_set_svm  <- subset(dataset_svm, split_svm == TRUE)
test_set_svm      <- subset(dataset_svm, split_svm == FALSE)

# SVM training ------------------------------------------------------------

dataset_svm %<>% replace(., is.na(.), 0)

tic("svmLinear training:")

classifier_svm <- train(
	x = dataset_svm[, 1:ncol(dataset_svm) - 1],
	y = dataset_svm$Churn,
 	method = "svmLinear"#,
	#metric = ifelse(is.factor(dataset_svm$Churn), "Accuracy", "RMSE"),
  #maximize = ifelse(metric == "RMSE", FALSE, TRUE)
)

toc()

# Predicting the Test set results ----------------------------------------

prob_pred_svm <- predict(
  classifier_svm,
  type = 'response',
  newdata = test_set_svm[-161]
)

y_pred_svm <- ifelse(prob_pred_svm > 0.5, 1, 0)

# Making the Confusion Matrix
cm_svm = table(test_set[, 161], y_pred_svm > 0.5)
# Results:
# True Negatives: 11573/11582 (99.92%)
# True Positives: 1/918 (0.11%)


# Performing the splitting to the rough dataset ----------------------------

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
