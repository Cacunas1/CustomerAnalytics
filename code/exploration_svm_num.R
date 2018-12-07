# Set-up environment ------------------------------------------------------

rm(list = ls())

library(caret)
library(dplyr)
library(magrittr)
library(readr)
library(tictoc)
library(tidyr)

# Importing the dataset --------------------------------------------------

orange_train <- read_tsv("./data/orange_small_train.data")
orange_test  <- read_tsv("./data/orange_small_test.data")
objective    <- read_csv("./data/KDD_CUP_2009_OBJECTIVE_COLUMNS_LABELS.csv")

# Functions --------------------------------------------------------------

not_all_na <- function(x) {!all(is.na(x))}

NA2mean <- function(x) replace(x, is.na(x), mean(x, na.rm = TRUE))

# Character to number -----------------------------------------------------

# Discard fully null columns
orange_train %<>% select_if(not_all_na)
orange_test  %<>% select_if(not_all_na)

# Save indexes of categorical/numerical variables
nums <- lapply(orange_train, is.numeric) %>% unlist()
cats <- lapply(orange_train, is.character) %>% unlist()

# Separate numerical predictors from categorical ones
train_pred_num <- orange_train[nums]
train_pred_cat <- orange_train[cats]
test_pred_num  <- orange_test[nums]
test_pred_cat  <- orange_test[cats]

# replace numerical predictor na's by the mean
train_pred_num %<>% mutate_all(funs(ifelse(is.na(.), mean(.,na.rm = T), .)))

# randomForest training ---------------------------------------------------

tic("svmLinear training:")

classifier_rf <- train(
  x = train_pred_num,
  y = objective$Churn,
  method = "svmLinear",
  verbose = T
  #metric = ifelse(is.factor(dataset_svm$Churn), "Accuracy", "RMSE"),
  #maximize = ifelse(metric == "RMSE", FALSE, TRUE)
)

toc()
