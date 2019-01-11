rm(list = ls())

library(dplyr)
library(caret)
library(pROC)

# Functions ---------------------------------------------------------------

NULLremover <- function(data,perc) {
  a <- lapply(data, function(x) sum(is.na(x))/length(x)) %>% unlist()
  ind <- which(a<perc)
  return(data[,ind])
}


SDremover <- function(data,perc) {
  a <- lapply(data, function(x) sd(as.numeric(x),na.rm=T)/
                (max(as.numeric(x),na.rm=T)-min(as.numeric(x),na.rm = T))) %>%
    unlist()
  ind <- which(a>perc)
  return(data[,ind])
}

#Function taking a column and replacing all categories with a Frequency less than
#thres with the category "Other" - in case of numeric variables no change.
categoryToOther <- function(col,thres) {
  check <- 0
  out <- col
  if(is.factor(col)) {
    t <- table(col)
    r <- which(t<thres)
    for(i in seq(1,max(1,length(r)),100)) {
      if(length(r)-i>=100) {
        out <- gsub(paste(names(r)[i:(i+99)],collapse="|"),"other",out) %>% factor()
      }
      else {
        out <- gsub(paste(names(r)[i:length(r)],collapse="|"),"other",out) %>% factor()
      }
    }
    check <- 1
  }
  cat(check) #just to see how far the process got
  return(out)
}
#for applying this to a dataframe df call
#df <- df %>% lapply(function(x) categoryToOther(x,50)) %>% data.frame()

#
binaryEncoding <- function(col) {
  if(is.factor(col)) {
    col_num <- col %>% as.numeric() %>% intToBits() %>%
      as.integer() %>% matrix(ncol=length(col)) %>% t() %>% data.frame()
    ind <- col_num %>% apply(2,function(x) all(x==0)) %>% {which(.==FALSE)} %>%
      max()
    return(col_num[,1:ind])
  }
  else return(col)
}

#####Program

# Load data ---------------------------------------------------------------

#setwd("E:/Users/Richard.Vogg/Documents/Customer Analytics/PoC_ML")

data <- read.table("data/orange_small_train.data.csv",sep=",",header=T)
labels <- read.table("data/labels.csv",sep=",",header=T)
data_orig <- data
data[data==""] <- NA


# Preprocessing with functions from Data review.R ----------------------------
data2 <- data %>%
  select_if(function(x) is.numeric(x) | nlevels(x)<5100) %>% #removes Var217,Var214,Var202,Var200
  select(-Var198,-Var220) %>% #as they are identical to Var222
  NULLremover(1) %>%
  SDremover(0) %>%  #until here is the standard cleaning
  mutate_if(is.numeric, funs(ifelse(is.na(.), mean(.,na.rm = T),.))) %>%
  mutate_if(is.factor, as.character) %>%
  mutate_if(is.character,funs(factor(ifelse(is.na(.), "missing", .)))) %>%
  lapply(function(x) categoryToOther(x,100)) %>% data.frame() %>%
  lapply(function(x) x <- binaryEncoding(x)) %>% data.frame()

#only for Random Forest without Binary Encoding
#data2 <- data2 %>% select_if(function(x) is.numeric(x) | nlevels(x)<50)


# Test Train Split --------------------------------------------------------
#################

set.seed(64)
ind <- sample(1:nrow(data2),0.8*(nrow(data2)))
train <- data2[ind,]
test <- data2[-ind,]

train_labels <- labels[ind,]
test_labels <- labels[-ind,]

# Sampling ----------------------------------------------------------------
##########

train_upsample <- downSample(x = train, y = factor(train_labels$Churn))


#Control for ML algorithms
##########################

fitControl <- trainControl(method = "CV", #Cross-validation
                           number = 3, #3-fold
                           verboseIter = TRUE, #Output while running
                           classProbs= TRUE, #needed for ROC
                           summaryFunction = twoClassSummary ) #needed for ROC

##############
# Churn prediction --------------------------------------------------------
##############


#Random forest
##############

rf_grid <- expand.grid(mtry=90,splitrule="gini",min.node.size=100)
set.seed(1)
rf <- train(x=train_upsample[,-ncol(train_upsample)],
            y=make.names(train_upsample$Class),
            method="ranger", metric="ROC",num.trees=500,
            trControl=fitControl,tuneGrid=rf_grid,importance = 'impurity')
rf_pred <- predict(rf,test,type="prob")
#table(rf_pred$X1>0.5,test_labels$Churn)


#Gradient boosting trees
########################

gbm_grid <- expand.grid(n.trees=seq(100,600), interaction.depth=6, shrinkage=0.01,
                       n.minobsinnode=10)
set.seed(1)
gbm <- train(x=train_upsample[,-ncol(train_upsample)],
            y=make.names(train_upsample$Class),
            method="gbm", metric="ROC",
            trControl=fitControl,tuneGrid=gbm_grid)
gbm_pred <- predict(gbm,test,type="prob")
#table(gbm_pred$X1>0.5,test_labels$Churn)


#Support vector machine
#######################

svm_grid <- expand.grid(C=1,sigma=c(0.001,0.002))
set.seed(1)
svm <- train(x=train_upsample[,-ncol(train_upsample)],
                    y=make.names(train_upsample$Class),
                    method = "svmRadial",
                    metric="ROC",
                    preProcess = c("center","scale"), #necessary for svm
                    trControl=fitControl,
                    tuneGrid=svm_grid)
svm_pred <- predict(svm,test,type="prob")
#table(svm_pred$X1>0.5,test_labels$Churn)


#Neural Network
################

nn_grid <- data.frame(size=5:10)
set.seed(1)
nn <- train(x=train_upsample[,-ncol(train_upsample)],
            y=make.names(train_upsample$Class),
            method="mlp", metric="ROC",
            trControl=fitControl,tuneGrid=nn_grid,
            preProcess = c("center","scale")
            )
nn_pred <- predict(nn,test,type="prob")
#table(nn_pred,test_labels$Churn)


#Logistic Regression
####################

set.seed(1)
lr <- train(x=train_upsample[,-ncol(train_upsample)],
            y=make.names(train_upsample$Class),
            method="glm", metric="ROC",
            trControl=fitControl#,tuneGrid=lr_grid
            #preProcess = c("center","scale")
)
lr_pred <- predict(lr,test,type="prob")
#table(lr_pred$X1>0.5,test_labels$Churn)


#Calculating the AUC

roc_rf <- roc(test_labels$Churn, rf_pred$X1)
auc(roc_rf)

roc_gbm <- roc(test_labels$Churn, gbm_pred$X1)
auc(roc_gbm)

roc_nn <- roc(test_labels$Churn, nn_pred$X1)
auc(roc_nn)

roc_svm <- roc(test_labels$Churn, svm_pred$X1)
auc(roc_svm)

roc_lr <- roc(test_labels$Churn, lr_pred$X1)
auc(roc_lr)

plot(roc_rf)
lines(roc_gbm,col="red")
lines(roc_nn,col="blue")
lines(roc_svm,col="green")
lines(roc_lr,col="orange")
