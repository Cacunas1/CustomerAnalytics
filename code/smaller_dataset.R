# Set-up environment ------------------------------------------------------

rm(list = ls())

library(dplyr)
library(magrittr)
library(readr)
#library(tictoc)
#library(tidyr)

# Functions --------------------------------------------------------------

not_all_na <- function(x) {!all(is.na(x))}

NA2mean <- function(x) replace(x, is.na(x), mean(x, na.rm = TRUE))

zero_var <- function(x) {length(unique(x)) > 1}

# Importing the dataset --------------------------------------------------

orange_train <- read_tsv("./data/orange_small_train.data")
orange_test  <- read_tsv("./data/orange_small_test.data")
objective    <- read_csv("./data/KDD_CUP_2009_OBJECTIVE_COLUMNS_LABELS.csv")

# Data transformation ----------------------------------------------------

# Objective as factor
objective$Churn[objective$Churn  == -1] <- 0
churn <- factor(objective$Churn, levels = c(0, 1))

# Discard fully null columns
orange_train %<>% select_if(not_all_na)
orange_test  %<>% select_if(not_all_na)

# Discard zero variance columns
orange_train %<>% select_if(zero_var)
orange_test  %<>% select_if(zero_var)

# bind response column
orange_train %<>% cbind(churn)

# random sample both classes for subseting
plus  <- churn == 1
minus <- churn == 0

orange_train_plus  <- orange_train[plus, ]
orange_train_minus <- orange_train[minus, ]

number_to_sample <- 2500

set.seed(69)

plus_rows  <- sample(nrow(orange_train_plus),  number_to_sample)
minus_rows <- sample(nrow(orange_train_minus), number_to_sample)

orange_train_plus_selected  <- orange_train_plus[plus_rows, ]
orange_train_minus_selected <- orange_train_minus[minus_rows, ]

trainingset <- rbind(orange_train_plus_selected, orange_train_minus_selected)

# save for testing
remainder <- orange_train[-plus_rows, ]
remainder <- remainder[-minus_rows, ]

write.csv(trainingset, "./output/trainingset.csv")
write.csv(remainder, "./output/remainder.csv")
