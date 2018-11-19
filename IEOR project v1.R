library(tm)
library(SnowballC)
library(MASS)
library(caTools)
library(plyr)
library(dplyr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(ggplot2)
library(rsample)
library(ranger)
library(h2o)

######## predicting Over counter names with only review + condition #########

drug = read.csv("DrugComUpdated.csv", stringsAsFactors=FALSE)
drug$drugName <- NULL

####### rows to remove?  #####
x <-count(drug,over_counter_name)
summary(x)
# min = 1
# 1st quartile = 24
# 2nd = 75
# 3rd = 222.5
# max (most frequent #) = 1306

####### NLP ##########
corpusCondition = Corpus(VectorSource(drug$condition))
corpusReview = Corpus(VectorSource(drug$review))

corpusCondition = tm_map(corpusCondition, tolower)
corpusReview = tm_map(corpusReview, tolower)

corpusCondition = tm_map(corpusCondition, removePunctuation)
corpusReview = tm_map(corpusReview, removePunctuation)

corpusReview = tm_map(corpusReview, removeNumbers)
corpusReview = tm_map(corpusReview, removeWords, stopwords("english"))
corpusReview = tm_map(corpusReview, removeWords, c("im"))

corpusCondition = tm_map(corpusCondition, stemDocument)
corpusReview = tm_map(corpusReview,stemDocument)

strwrap(corpusReview[[1]])
strwrap(corpusCondition[[1]])

CondFreq = DocumentTermMatrix(corpusCondition)
RevFreq = DocumentTermMatrix(corpusReview)

# this sparsity leads to 165 total variables. Should we cut down more? 
sparseCond = removeSparseTerms(CondFreq, 0.99)
sparseRev = removeSparseTerms(RevFreq, 0.95)

CondData = as.data.frame(as.matrix(sparseCond))
RevData = as.data.frame(as.matrix(sparseRev))

drugData<-cbind(CondData,RevData)
colnames(drugData) = make.names(colnames(drugData))
drugData$over_counter_name = drug$over_counter_name

for (i in 1:ncol(drugData))
{names(drugData) <- make.names(names(drugData), unique = TRUE) }

set.seed(123) 
spl = sample.split(drugData$over_counter_name, SplitRatio = 0.7)
drug.train = drugData %>% filter(spl == TRUE)
drug.test = drugData %>% filter(spl == FALSE)

tableAccuracy <- function(test, pred) {
  t = table(test, pred)
  a = sum(diag(t))/length(test)
  return(a)
}


##### LDA ####
lda.mod = lda(over_counter_name ~ ., data = drug.train)
# I got this warning: In lda.default(x, grouping, ...) : variables are collinear

pred.lda = predict(lda.mod, newdata = drug.test)$class
table(drug.test$over_counter_name, pred.lda)
tableAccuracy(drug.test$over_counter_name, pred.lda)
#accuracy:  0.01690219

###### Cross validated RF ######
set.seed(100)
train.rf1 = train(over_counter_name ~ .,
                 data = drug.train,
                 method = "rf",
                 tuneGrid = data.frame(mtry = 1:16),
                 trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE))
train.rf1
#mtry = 11
# r gave me errors/warnings from fold 3 mtry = 1 to fold 4 mtry = 16 
# saying "the following model fit failed for Fold3: mtry= 1 Error in randomForest.default(x, y, mtry = param$mtry, ...) : Can't have empty classes in y.

train.rf1$results
mod.rf1 = train.rf1$finalModel
pred.rf1 = predict(mod.rf1, newdata = drug.test)
table(drug.test$over_counter_name, pred.rf1)
tableAccuracy(drug.test$over_counter_name, pred.rf1)
#0.02349681

###### default RF with ranger package ######
ranger.mod <- ranger(over_counter_name ~ ., data = drug.train)
ranger.pred <- predict(ranger.mod, data = drug.test)
table(drug.test$over_counter_name, ranger.pred$predictions)
tableAccuracy(drug.test$over_counter_name, ranger.pred$predictions)
#0.02294264

###### Cross Validated RF with ranger package ######
set.seed(100)
#tuning parameters below. I'm not sure about the splitrule and min.node.size, so I set them based off of the ranger documentation's default recommendations 
tgrid <- expand.grid(
  .mtry = 16:50,
  .splitrule = "gini",
  .min.node.size = 1
)

train.rf = train(over_counter_name ~ .,
                 data = drug.train,
                 method = "ranger",
                 tuneGrid = tgrid,
                 trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE))
train.rf

### when trained on mtry 1:16 ###
# mtry = 12
### when trained on mtry 16:50 ###
# mtry = 16

# I got the following warning messages for avita, proscar, vasotec on both rf methods (normal and ranger rf): 
#    "Dropped unused factor level(s) in dependent variable"


mod.rf = train.rf$finalModel
pred.rf = predict(mod.rf, data = drug.test)
table(drug.test$over_counter_name, pred.rf$predictions)
tableAccuracy(drug.test$over_counter_name, pred.rf$predictions)

######## predicting Over counter names without review #########





