library(xgboost)
library(Matrix)

set.seed(1234)

train <- read.csv("/home/rishikesh/Dev/Python/Mckinsey/train.csv")
test  <- read.csv("/home/rishikesh/Dev/Python/Mckinsey/test.csv")

# xtrain and xtest are just clean data done in python by following code written in comment below

########### Python Code to Data preprocessing use to create xtrain.csv and xtest.csv ################################
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import preprocessing
# 
# train_df   = pd.read_csv('/home/rishikesh/Dev/Python/Mckinsey/train.csv')
# test_df  = pd.read_csv('/home/rishikesh/Dev/Python/Mckinsey/test.csv')
# train_df.head()
# test_df["Email_ID"].head()
# test_df["Email_ID"].head()
# 
# 
# for feat in train_df.columns:
#   if train_df[feat].dtype == 'object':
#     lbl = preprocessing.LabelEncoder()
#     lbl.fit(np.unique(list(train_df[feat].values) + list(test_df[feat].values)))
#     train_df[feat]   = lbl.transform(list(train_df[feat].values))
#     test_df[feat]  = lbl.transform(list(test_df[feat].values))
# 
# # remove constant columns
# remove = []
# for col in train_df.columns:
#   if train_df[col].std() == 0:
#     remove.append(col)
# 
# train_df.drop(remove, axis=1, inplace=True)
# test_df.drop(remove, axis=1, inplace=True)
# 
# # remove duplicated columns
# remove = []
# c = train_df.columns
# for i in range(len(c)-1):
#   v = train_df[c[i]].values
#   for j in range(i+1,len(c)):
#     if np.array_equal(v,train_df[c[j]].values):
#     remove.append(c[j])
# 
# train_df.drop(remove, axis=1, inplace=True)
# test_df.drop(remove, axis=1, inplace=True)
# 
# 
# for feat in train_df.columns:
#   if train_df[feat].dtype == 'float64':
#   train_df[feat][np.isnan(train_df[feat])] = train_df[feat].mean()
#   test_df[feat][np.isnan(test_df[feat])] = test_df[feat].mean()
# 
# elif train_df[feat].dtype == 'object':
#   train_df[feat][train_df[feat] != train_df[feat]] = train_df[feat].value_counts().index[0]
#   test_df[feat][test_df[feat] != test_df[feat]] = test_df[feat].value_counts().index[0]
#
#
# X_train = train_df.drop(["Email_Status","Email_ID"],axis=1)
# Y_train = train_df["Email_Status"]
# X_test  = test_df.drop("Email_ID",axis=1).copy()
#
#
# X_train.to_csv('xtrain.csv', index=False)
# X_test.to_csv('xtest.csv', index=False)
######################################################################################

xtrain <- read.csv("/home/rishikesh/Dev/Python/Mckinsey/xtrain.csv")
xtest  <- read.csv("/home/rishikesh/Dev/Python/Mckinsey/xtest.csv")
##### Removing IDs
train$Email_ID <- NULL
test.id <- test$Email_ID
test$Email_ID <- NULL

##### Extracting TARGET
train.y <- train$Email_Status
xtrain$Email_Status <- NULL

subSet1<-xtrain[1:34176,]
subSet2<-xtrain[34177:68353,]
target1<-train.y[1:34176]
target2<-train.y[34177:68353]

##################################################################################################################
subSet1<-xtrain[1:34176,]
subSet2<-xtrain[34177:68353,]
target1<-train.y[1:34176]
target2<-train.y[34177:68353]
dat1 <- xgb.DMatrix(as.matrix(subSet1), label = target1)
dtest1 <- as.matrix(subSet2)
clf1 <- NULL
preds1<-NULL
acc1<-0
param <- list(  objective           = "multi:softmax", 
                booster             = "gbtree",
                eval_metric         = "merror",
                num_class           = 3,
                eta                 = 1,
                max_depth           = 5,
                subsample           = 0.6815,
                colsample_bytree    = 0.701
)

clf1 <- xgboost(   params              = param, 
                  data                = dat1,
                  verbose             = 1,
                  nrounds             = 5,
                  nthread              = 8,
                  maximize            = FALSE
)
preds1 <- predict(clf1, dtest1)
##################################################################################################################
count <- 0
i<-0
for(i in c(1:34177)){
  
  if(preds1[i]!=target2[i])
  {
    count<-count+1
  }
}
print(count)
acc1<-1-(count/34177)
##################################################################################################################
dat2 <- xgb.DMatrix(as.matrix(subSet2), label = target2)
dtest2 <- as.matrix(subSet1)


clf2 <- xgboost(   params              = param, 
                   data                = dat2,
                   nrounds             = 5, 
                   verbose             = 1,
                   nthread              = 8,
                   maximize            = FALSE
)
preds2 <- predict(clf2, dtest2)
##################################################################################################################

count <- 0
for(i in c(1:34176)){
  
  if(preds2[i]!=target1[i])
  {
    count<-count+1
  }
}
acc2<-1-(count/34176)
#######################################################################################################################
count <- 0
t5<-c(preds2,preds1)
for(i in c(1:68353)){
  
  if(train.y[i]!=t5[i])
  {
    count<-count+1
  }
}
acc3<-1-(count/68353)


########################################################################################################################
dat <- xgb.DMatrix(as.matrix(xtrain), label = train.y)
dtest <- as.matrix(xtest)


clf <- xgboost(   params              = param, 
                    data                = dat,
                    nrounds             = 5, 
                    verbose             = 1,
                    nthread              = 8,
                    maximize            = FALSE
)

predst5 <- predict(clf, dtest)
############################################################################################################################

###########################################################################################################################
predf<-cbind(predst1,predst2,predst3,predst4,predst5)
tf<-cbind(t1,t2,t3,t4,t5)

###########################################################################################################################

dat <- xgb.DMatrix(tf, label = train.y)
dtest <- as.matrix(predf)
clf<-NULL
param<-NULL
preds<-NULL
param <- list(  objective           = "multi:softmax", 
                booster             = "gbtree",
                eval_metric         = "merror",
                num_class           = 3,
                eta                 = 0.15,
                max_depth           = 5,
                subsample           = 0.6815,
                colsample_bytree    = 0.701
)

clf <- xgboost(   params              = param, 
                   data                = dat,
                   nrounds             = 116, 
                   verbose             = 1,
                   nthread              = 8,
                   maximize            = FALSE
)
preds <- predict(clf, dtest)






######################################################################################################################################










#################################################################################################################################


submission <- data.frame(Email_ID=test.id, Email_Status=preds)
cat("saving the submission file\n")
write.csv(submission, "/home/rishikesh/Dev/Python/Mckinsey/Rsubmission.csv", row.names = F)