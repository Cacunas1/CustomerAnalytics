# Set-up environment ------------------------------------------------------

rm(list = ls())

library(caret)
library(doParallel)
library(dplyr)
library(magrittr)
library(readr)
library(tictoc)
library(tidyr)

# Importing the dataset --------------------------------------------------

train_pred_num <- read_csv("./output/train_pred_num.csv")
remainder_num <- read_csv("./output/remainder_num.csv")

#train_pred_num$churn %<>% as.factor()
#remainder_num$churn %<>% as.factor()
train_churn <- train_pred_num$churn
remainder_churn <- remainder_num$churn

train_churn[train_churn == 2] <- "retained"
train_churn[train_churn == 1] <- "churned"

# SVM Linear training ---------------------------------------------------

set.seed(69)

fit_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  allowParallel = TRUE,
  savePredictions = TRUE
)

nlevels(train_pred_num$churn)
class(train_pred_num)

sink("./output/svmlinear_5000.out")

tic("svmLinear training:")

cl <- makePSOCKcluster(4)
registerDoParallel(cl)

classifier_svm <- train(
  x = train_pred_num[1:174],
  y = train_pred_num$churn,
  method = "svmLinear",
  trControl = fit_control,
  verbose = T
  #trControl = trainControl(method = "repeatedcv", repeats = 5)
  #metric = ifelse(is.factor(dataset_svm$Churn), "Accuracy", "RMSE"),
  #maximize = ifelse(metric == "RMSE", FALSE, TRUE)
)

stopCluster(cl)

toc()

out_svm <- predict(classifier_svm, remainder_num[-churn])
cm_svm  <- confusionMatrix(out_svm, remainder$churn)

print(cm_svm)

sink()
