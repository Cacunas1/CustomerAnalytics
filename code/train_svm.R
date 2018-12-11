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
remainder <- read_csv("./output/remainder.csv")

# training ----------------------------------------------------------------

# Save indexes of categorical/numerical variables
nums <- lapply(trainingset[1:ncol(trainingset) - 1], is.numeric) %>% unlist()
cats <- lapply(trainingset[1:ncol(trainingset) - 1], is.character) %>% unlist()

# Separate numerical predictors from categorical ones
train_pred_num <- trainingset[c(nums, F)]
train_pred_cat <- trainingset[c(cats, F)]
test_pred_num  <- trainingset[c(nums, F)]
test_pred_cat  <- trainingset[c(cats, F)]

# replace numerical predictor na's by the mean
train_pred_num %<>% mutate_all(funs(ifelse(is.na(.), mean(.,na.rm = T), .)))

trainingset$churn %<>% as.factor()

# SVM Linear training ---------------------------------------------------

sink("./output/svmlinear_5000.out")

tic("svmLinear training:")

cl <- makePSOCKcluster(4)
registerDoParallel(cl)

classifier_svm <- train(
  x = train_pred_num,
  y = trainingset$churn,
  method = "svmLinear",
  verbose = T,
  allowParallel = T
  #metric = ifelse(is.factor(dataset_svm$Churn), "Accuracy", "RMSE"),
  #maximize = ifelse(metric == "RMSE", FALSE, TRUE)
)

stopCluster(cl)

toc()

out_svm <- predict(classifier_svm, remainder[1:ncol(remainder) - 1])
cm_svm <- confusionMatrix(out_svm, remainder$churn)
print(cm_svm)
sink()
