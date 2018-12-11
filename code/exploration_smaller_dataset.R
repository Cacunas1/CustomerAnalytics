# Set-up environment ------------------------------------------------------

rm(list = ls())

library(caret)
library(dplyr)
library(magrittr)
library(readr)
library(tictoc)
library(tidyr)

# Functions --------------------------------------------------------------

not_all_na <- function(x) {!all(is.na(x))}

NA2mean <- function(x) replace(x, is.na(x), mean(x, na.rm = TRUE))

# Importing the dataset --------------------------------------------------

orange_train <- read_tsv("./data/orange_small_train.data")
orange_test  <- read_tsv("./data/orange_small_test.data")
objective    <- read_csv("./data/KDD_CUP_2009_OBJECTIVE_COLUMNS_LABELS.csv")

# Character to number -----------------------------------------------------

objective$Churn[objective$Churn  == -1] <- 0
churn <- factor(objective$Churn, levels = c(0, 1))

plus  <- churn == 1
minus <- churn == 0

class_plus  <- sum(plus)
class_minus <- sum(minus)

# Discard fully null columns
orange_train %<>% select_if(not_all_na)
orange_test  %<>% select_if(not_all_na)

orange_train %<>% cbind(churn)

orange_train_plus  <- orange_train[plus, ]
orange_train_minus <- orange_train[minus, ]

number_to_sample <- 2500

set.seed(69)

orange_train_plus_selected <-
  orange_train_plus[sample(nrow(orange_train_plus), number_to_sample), ]

orange_train_minus_selected <-
  orange_train_minus[sample(nrow(orange_train_minus), number_to_sample), ]

trainingset <- rbind(orange_train_plus_selected, orange_train_minus_selected)

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

# predictor training ---------------------------------------------------

tic("svmLinear training:")

classifier_svm <- train(
  x = train_pred_num,
  y = trainingset$churn,
  method = "svmLinear",
  verbose = T
  #metric = ifelse(is.factor(dataset_svm$Churn), "Accuracy", "RMSE"),
  #maximize = ifelse(metric == "RMSE", FALSE, TRUE)
)

toc()
