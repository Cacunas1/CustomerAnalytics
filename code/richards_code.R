#library(reshape2)
library(caret)
library(dplyr)
library(ggplot2)
library(randomForest)
library(stringdist)

#setwd("E:/Users/Richard.Vogg/Documents/Customer Analytics/PoC_ML")
data <- read.table("./data/orange_small_train.data.csv",sep=",",header=T)
labels <- read.table("./data/labels.csv",sep=",",header=T)
data_orig <- data
data[data==""] <- NA

#############
#Functions
#############

NULLcounter <- function(perc) {
  a <- lapply(data, function(x) sum(is.na(x))/length(x)) %>% unlist()
  return(sum(a<perc))
}

NULLremover <- function(perc,data) {
  a <- lapply(data, function(x) sum(is.na(x))/length(x)) %>% unlist()
  ind <- which(a<perc)
  data <- data[,ind]
  return(data)
}

SDcounter <- function(perc) {
  a <- lapply(data, function(x) sd(as.numeric(x),na.rm=T)/
                (max(as.numeric(x),na.rm=T)-min(as.numeric(x),na.rm = T))) %>%
    unlist()
  a[is.na(a)] <- 0
  return(sum(a<perc))
}

SDremover <- function(perc,data) {
  a <- lapply(data, function(x) sd(as.numeric(x),na.rm=T)/
                (max(as.numeric(x),na.rm=T)-min(as.numeric(x),na.rm = T))) %>%
    unlist()
  ind <- which(a>perc)
  data <- data[,ind]
  return(data)
}

categoryToOther <- function(col,thres) {
  out <- col
  if(is.factor(col)) {
    t <- table(col)
    r <- which(t<thres)
    for(i in seq(1,max(1,length(r)),100)) {
      if(length(r)-i>=100) {
        out <- factor(gsub(paste(names(r)[i:(i+99)],collapse="|"),"other",out))
      }
      else {
        out <- factor(gsub(paste(names(r)[i:length(r)],collapse="|"),"other",out))
      }
    }
  }
  return(out)
}

#I use the above functions for the preprocessing:

data2 <- data %>% {NULLremover(0.99,.)} %>% SDremover(0,.) %>%
  mutate_if(is.numeric, funs(ifelse(is.na(.), mean(.,na.rm = T),.))) %>%
  mutate_if(is.factor, as.character) %>%
  mutate_if(is.character,funs(factor(ifelse(is.na(.), "missing", .)))) %>%
  lapply(function(x) categoryToOther(x,50)) %>% data.frame()

data2 <- data2 %>% select_if(function(x) is.numeric(x) | nlevels(x)<50)

#Split into train and test; as the dataset is imbalanced I hoped to get better results with upsampling the churn cases, to have 50:50 in the trainset.

ind <- sample(1:nrow(data2),0.8*(nrow(data2)))
train <- data2[ind,]
test <- data2[-ind,]

train_labels <- labels[ind,]
test_labels <- labels[-ind,]

train_upsample <- upSample(x = train, y = factor(train_labels$Churn))

#And finally the Random forest, I fixed most of the parameters which makes it run less times:

fitControl <- trainControl(method = "CV",
                           number = 3,
                           verboseIter = TRUE,
                           classProbs= TRUE,
                           summaryFunction = twoClassSummary)

rf_grid <- data.frame(mtry=25,splitrule="gini",min.node.size=2)

rf <- train(x=train_upsample[,-ncol(train_upsample)],
            y=make.names(train_upsample$Class),
            method="ranger", metric="ROC",num.trees=200,
            trControl=fitControl,tuneGrid=rf_grid)
rf_pred <- predict(rf,test)
table(rf_pred,test_labels$Churn)
