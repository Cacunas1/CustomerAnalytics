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

trainingset <- read_csv("./output/trainingset.csv")
#remainder <- read_csv("./output/remainder.csv")

# training ----------------------------------------------------------------

# Save indexes of categorical/numerical variables
nums <- lapply(trainingset[1:ncol(trainingset) - 1], is.numeric) %>% unlist()
cats <- lapply(trainingset[1:ncol(trainingset) - 1], is.character) %>% unlist()

# Separate numerical predictors from categorical ones
train_pred_num <- trainingset[nums]
train_pred_cat <- trainingset[cats]
test_pred_num  <- trainingset[nums]
test_pred_cat  <- trainingset[cats]

# replace numerical predictor na's by the mean
train_pred_num %<>% mutate_all(funs(ifelse(is.na(.), mean(.,na.rm = T), .)))

# SVM Linear training ---------------------------------------------------

cl <- makePSOCKcluster(4)
registerDoParallel(cl)

sink("./output/svmlinear_5000.out")

tic("svmLinear training:")

classifier_svm <- train(
  x = train_pred_num,
  y = objective$Churn,
  method = "svmLinear",
  verbose = T
  #metric = ifelse(is.factor(dataset_svm$Churn), "Accuracy", "RMSE"),
  #maximize = ifelse(metric == "RMSE", FALSE, TRUE)
)

toc()

sink()

stopCluster(cl)