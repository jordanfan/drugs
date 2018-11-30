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
library(softImpute)


drug = read.csv("drugs_final1.csv", stringsAsFactors=FALSE, na.strings = c("", "NA"))
drug$drugName <- NULL

####### removing rows where over_counter_name has frequency <=50 #####
sorted_counts = as.data.frame(sort(table(drug$over_counter_name)))
keep_drugs = sorted_counts$Var1[106:nrow(sorted_counts)] 
drug = drug[drug$over_counter_name %in% keep_drugs,]

##### No clustering #####
######## predicting Over counter names with only review + condition #########
####### NLP ##########
corpusCondition = Corpus(VectorSource(drug$condition))
corpusCondition = tm_map(corpusCondition, tolower)

corpusReview = Corpus(VectorSource(drug$review))
corpusReview = tm_map(corpusReview, tolower)
corpusReview = tm_map(corpusReview, function(x) gsub("[^a-z0-9]", " ", x))
#corpusCondition = tm_map(corpusCondition, removePunctuation)
#corpusReview = tm_map(corpusReview, removePunctuation)
corpusReview = tm_map(corpusReview, removeNumbers)
exception = c("not", "no", "few", "reduc", "immedi", "didnt", "most", "quick", "never", "littl", "well")
my_stopwords = setdiff(stopwords("english"), exception)
corpusReview = tm_map(corpusReview, removeWords, c(my_stopwords, "im"))
corpusReview = tm_map(corpusReview,stemDocument)

strwrap(corpusReview[[1]])
strwrap(corpusCondition[[1]])

CondFreq = DocumentTermMatrix(corpusCondition)
RevFreq = DocumentTermMatrix(corpusReview)

sparseCond = removeSparseTerms(CondFreq, 0.995)
sparseRev = removeSparseTerms(RevFreq, 0.90)

CondData = as.data.frame(as.matrix(sparseCond))
RevData = as.data.frame(as.matrix(sparseRev))

drugData<-cbind(CondData,RevData)
colnames(drugData) = make.names(colnames(drugData))
drugData$over_counter_name = drug$over_counter_name
#drugData$condition = as.factor(drug$condition)
drugData$effectiveness = as.factor(drug$effectiveness)
drugData$sideEffects = as.factor(drug$sideEffects)
drugData$price = as.numeric(drug$price)
drugData$rating = as.numeric(drug$rating)
drugData$usefulCount = as.numeric(drug$usefulCount)
drugData$sentiment = as.numeric(drug$sentiment)

for (i in 1:ncol(drugData))
{names(drugData) <- make.names(names(drugData), unique = TRUE) }

##### training ########

set.seed(123) 
spl = sample.split(drugData$over_counter_name, SplitRatio = 0.7)
drug.train = drugData %>% filter(spl == TRUE)
drug.test = drugData %>% filter(spl == FALSE)

tableAccuracy <- function(test, pred) {
  t = table(test, pred)
  a = sum(diag(t))/length(test)
  return(a)
}


drug.train.mm.nlp = as.data.frame(model.matrix(over_counter_name ~ . -1 -effectiveness -sideEffects -price -rating -usefulCount -sentiment, 
                                         data=drug.train))
drug.train.mm.all = as.data.frame(model.matrix(over_counter_name ~ . -1, 
                                               data=drug.train))
drug.test.mm.nlp = as.data.frame(model.matrix(over_counter_name ~ . -1 -effectiveness -sideEffects -price -rating -usefulCount -sentiment, 
                                          data=drug.test))
drug.test.mm.all = as.data.frame(model.matrix(over_counter_name ~ . -1, 
                                          data=drug.test))

set.seed(100)
#tuning parameters below. 
#mtry should probably be around sqrt(# of features)
tgrid <- expand.grid(
  .mtry = 1:20,
  .splitrule = "gini",
  .min.node.size = 1
)

###################    Cross Validated NLP only    ####################
#took 2 hours
train.rf.nlp = train(over_counter_name ~ . -effectiveness -sideEffects -price -rating -usefulCount -sentiment,
                  data = drug.train,
                  method = "ranger",
                  tuneGrid = tgrid,
                  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE))

# Testing mtry 1:20
# (sqrt(p)) is around 11-12
# CV mtry = 10

mod.rf.nlp = train.rf.nlp$finalModel 
pred.rf.nlp = predict(mod.rf.nlp, data = drug.test.mm.nlp)
table(drug.test$over_counter_name, pred.rf.nlp$predictions)
tableAccuracy(drug.test$over_counter_name, pred.rf.nlp$predictions)
# accuracy =  0.3115581

###################     CV - All     ########################
# also took around 2 hours
train.rf.all = train(over_counter_name ~ .,
                  data = drug.train,
                  method = "ranger",
                  tuneGrid = tgrid,
                  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE))

# Testing mtry 1:20
# (sqrt(p)) is around 11-12
# mtry = 20

mod.rf.all = train.rf.all$finalModel
pred.rf.all = predict(mod.rf.all, data = drug.test.mm.all)
tableAccuracy(drug.test$over_counter_name, pred.rf.all$predictions)
#0.1388026

#################     Default Ranger - NLP    #####################
#took a couple minutes
ranger.nlp = ranger(over_counter_name ~ .-effectiveness -sideEffects -price -rating -usefulCount -sentiment,
                 data = drug.train, mtry = 10, verbose = TRUE)

pred.rg.nlp = predict(ranger.nlp, data = drug.test)
tableAccuracy(drug.test$over_counter_name, pred.rg.nlp$predictions)
# accuracy = 0.3134932
#  #  #
ranger.nlp.default = ranger(over_counter_name ~ .-effectiveness -sideEffects -price -rating -usefulCount -sentiment,
                    data = drug.train, verbose = TRUE)

pred.rg.nlp.d = predict(ranger.nlp.default, data = drug.test)
tableAccuracy(drug.test$over_counter_name, pred.rg.nlp.d$predictions)
# accuracy = 0.3122618

################   Default Ranger - All     ######################
#takes like 2-3 minutes
ranger.all = ranger(over_counter_name ~ .,
                 data = drug.train, mtry = 20, verbose = TRUE)

pred.rg.all = predict(ranger.all, data = drug.test)
tableAccuracy(drug.test$over_counter_name, pred.rg.all$predictions)
# accuracy = 0.9121562
#  #  #

ranger.all.default = ranger(over_counter_name ~ .,
                    data = drug.train, verbose = TRUE)

pred.rg.all.d = predict(ranger.all.default, data = drug.test)
tableAccuracy(drug.test$over_counter_name, pred.rg.all.d$predictions)
#accuracy = 0.7733537

#######################################

