#install the package
install.packages("caret")
install.packages("skimr")
install.packages("DataExplorer")
install.packages("pROC")
install.packages("xgboost")
install.packages("Matrix")
install.packages("rpart")
install.packages("rpart.plot")

#library the package
library(caret)
library(skimr)
library(DataExplorer)
library(pROC)
library(xgboost)
library(Matrix)
library(rpart)
library(rpart.plot)

credit <- read.csv("https://www.dropbox.com/s/roxpeue7p78a8lc/creditdata.csv?dl=1")
# credit <- read.csv("data.csv")

# Data exploration
skimr::skim(credit)
DataExplorer::plot_bar(credit, ncol = 3)
DataExplorer::plot_histogram(credit, ncol = 3)

#Data preprocessing
#Delete empty data
credit <- na.omit(credit)
#No missing data!

#Convert English classified data into numerical values
credit1 <- credit
credit1$good_bad <- gsub("Good",1,credit1$good_bad)
credit1$good_bad <- gsub("Bad",0,credit1$good_bad)

for (i in 1:21){
  credit1[,i] <- gsub("A",1,credit1[,i])
  credit1[,i] <- gsub("B",2,credit1[,i])
  credit1[,i] <- gsub("C",3,credit1[,i])
  credit1[,i] <- gsub("D",4,credit1[,i])
  credit1[,i] <- gsub("E",5,credit1[,i])
  credit1[,i] <- gsub("F",6,credit1[,i])
  credit1[,i] <- gsub("G",7,credit1[,i])
  credit1[,i] <- gsub("H",8,credit1[,i])
  credit1[,i] <- gsub("I",9,credit1[,i])
  credit1[,i] <- gsub("J",10,credit1[,i])
}

#Data normalization
for (i in 1:20){
  credit1[,i] <- as.numeric(credit1[,i])
  credit1[,i] <- scale(credit1[,i])
}
credit1[,21] <- as.numeric(credit1[,21])
head(credit1)

#Classification modeling
#logistic
#split 
LR = glm(good_bad ~ .,
         data=credit1,
         family=binomial(link="logit"))

summary(LR)

LR2 = glm(good_bad ~ checking + duration + history + amount +
            savings + employed + instalp + marital + coapp + 
            property + other + foreign,
          data=credit1,
          family=binomial(link="logit"))
summary(LR2)

#split
set.seed(1234)
sub <- sample(nrow(credit1),nrow(credit1)*0.7)
train_data <- credit1[sub,]
test_data <- credit1[-sub,]
LR1 = glm(good_bad ~ .,
         data=train_data,
         family=binomial(link="logit"))
predict_=predict.glm(LR1,type="response",newdata=test_data)
predict=ifelse(predict_>0.5,1,0)
logistic_roc <- roc(test_data$good_bad,as.numeric(predict),ci=TRUE,pi = TRUE)

#choose the feature
LR2 = glm(good_bad ~ checking + duration + history + amount +
            savings + employed + instalp + marital + coapp + 
            property + other + foreign,
          data=train_data,
          family=binomial(link="logit"))
predict_LR2=predict.glm(LR2,type="response",newdata=test_data)
predictLR2=ifelse(predict_LR2>0.5,1,0)
logistic_roc_LR2 <- roc(test_data$good_bad,as.numeric(predictLR2),ci=TRUE,pi = TRUE)

par(mfrow=c(1,2))
plot(logistic_roc, print.auc=TRUE, auc.polygon=TRUE,legacy.axes=TRUE, 
     grid.col=c("green", "red"), max.auc.polygon=TRUE, 
     auc.polygon.col="skyblue", print.thres=TRUE,
     main = "Logistic(All Var) of ROC")
plot(logistic_roc_LR2, print.auc=TRUE, auc.polygon=TRUE,legacy.axes=TRUE, 
     grid.col=c("green", "red"), max.auc.polygon=TRUE, 
     auc.polygon.col="skyblue", print.thres=TRUE,
     main = "Logistic(P<0.05 Var) of ROC")
par(mfrow=c(1,1))

pre_LR2 = round(predict(LR2,newdata = test_data))
predictLR2=ifelse(predict_LR2>0.5,1,0)
LR2table = table(test_data$good_bad,predictLR2,dnn=c("true","pre"))
LR2table
print(paste0("Accuracy of LR2(P<0.05):",round((LR2table[1,1]+LR2table[2,2])/sum(LR2table),4)*100,"%"))

#Xgboost
traindata1 <- data.matrix(train_data[,c(1:20)]) # Convert arguments to matrices
traindata2 <- Matrix(traindata1,sparse=T) # 
traindata3 <- as.numeric(as.character(train_data[,21])) # Using the matrix function, set the sparse parameter to true and convert it into a sparse matrix
traindata4 <- list(data=traindata2,label=traindata3) # Splice the independent variable and dependent variable into a list
dtrain <- xgb.DMatrix(data = traindata4$data, label = traindata4$label) # The xgb.dmatrix object required to construct the model is a sparse matrix

testset1 <- data.matrix(test_data[,c(1:20)]) 
testset2 <- Matrix(testset1,sparse=T)
testset3 <- as.numeric(as.character(test_data[,21]))
testset4 <- list(data=testset2,label=testset3) 
dtest <- xgb.DMatrix(data = testset4$data, label = testset4$label) 

param <- list(max_depth=20, eta=1, objective='binary:logistic') # Define model parameters
nround = 1000
bst = xgb.train(params = param, data = dtrain, nrounds = nround, nthread = 2) # Constructing xgboost model
names <- dimnames(data.matrix(train_data[,c(1:20)]))[[2]] # Gets the real name of the feature
importance_matrix <- xgb.importance(names,model=bst) # Calculate variable importance
xgb.plot.importance(importance_matrix[,])
#XGB confusion matrix
pre_xgb = round(predict(bst,newdata = dtest))
xgbtable = table(test_data$good_bad,pre_xgb,dnn=c("true","pre"))
xgbtable
print(paste0("Accuracy of XGboost:",round((xgbtable[1,1]+xgbtable[2,2])/sum(xgbtable),4)*100,"%"))


#imporvement
depth_list = seq(from=1, to=20, by=1)
eta_list = seq(from=0.6, to=1, by=0.1)
max_depth_record = NULL
eta_record = NULL
acc = NULL
a = 1
for (i in depth_list) {
  for (j in eta_list){
    param <- list(max_depth=i, eta=j, objective='binary:logistic')
    nround = 100
    bst = xgb.train(params = param, data = dtrain, nrounds = nround, nthread = 2) 
    pre_xgb = round(predict(bst,newdata = dtest))
    xgbtable = table(test_data$good_bad,pre_xgb,dnn=c("true","pre"))
    # print(paste0("depth:",i," ,eta:",j," ,Accuracy:",round((xgbtable[1,1]+xgbtable[2,2])/sum(xgbtable),4)*100,"%"))
    max_depth_record[a] = i
    eta_record[a] = j
    acc[a] = round((xgbtable[1,1]+xgbtable[2,2])/sum(xgbtable),4)*100
    a = a + 1
  }
}
xgb_record <- as.data.frame(max_depth_record)
xgb_record$eta_record <- eta_record
xgb_record$accuracy <- acc
xgb_record <- xgb_record[order(xgb_record$accuracy,decreasing = TRUE),]


dtree<-rpart(good_bad~.,data=train_data,method="class", parms=list(split="information"))
printcp(dtree)

tree<-prune(dtree,cp=dtree$cptable[which.min(dtree$cptable[,"xerror"]),"CP"])

rpart.plot(dtree,branch=1,type=2, fallen.leaves=T,cex=0.65, sub="Before pruning")
rpart.plot(tree,branch=1, type=4,fallen.leaves=T,cex=0.7, sub="After pruning")
predtree<-predict(tree,newdata=test_data,type="class")   #Prediction using prediction set
dtable = table(test_data$good_bad,predtree,dnn=c("real","predict"))    #Output confusion matrix
print(paste0("Accuracy of Decision Tree:",round((dtable[1,1]+dtable[2,2])/sum(dtable),4)*100,"%"))
