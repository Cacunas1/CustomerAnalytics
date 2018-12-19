# Set-up environment ------------------------------------------------------

rm(list = ls())

library(caret)
library(doParallel)
library(dplyr)
library(magrittr)
library(ranger)
library(readr)
library(tictoc)
library(tidyr)

# Importing the dataset --------------------------------------------------

trainingset <- read_csv("./output/trainingset.csv")
remainder   <- read_csv("./output/remainder.csv")

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

# Random Forest training ---------------------------------------------------

sink("./output/svmlinear_5000.out")

tic("Random Forest training:")

cl <- makePSOCKcluster(4)
registerDoParallel(cl)

classifier_rf <- train(
  x = train_pred_num,
  y = trainingset$churn,
  method = "ranger",
  verbose = T,
  allowParallel = T,
  ntrees = 100
  #metric = ifelse(is.factor(dataset_rf$Churn), "Accuracy", "RMSE"),
  #maximize = ifelse(metric == "RMSE", FALSE, TRUE)
)

stopCluster(cl)

toc()

out_rf <- predict(classifier_rf, remainder[1:ncol(remainder) - 1])
cm_rf <- confusionMatrix(out_rf, remainder$churn)
print(cm_rf)
sink()
