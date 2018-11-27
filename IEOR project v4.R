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


drug = read.csv("drugs_final.csv", stringsAsFactors=FALSE, na.strings = c("", "NA"))
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
drugData$condition = as.factor(drug$condition)
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

ggplot(train.rf$results, aes(x = mtry, y = Accuracy)) + geom_point(size = 2) + geom_line() + 
  ylab("CV Accuracy") + theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

###### default RF with ranger package ######
#ranger.mod <- ranger(over_counter_name ~ ., data = drug.train)
#ranger.pred <- predict(ranger.mod, data = drug.test)
#table(drug.test$over_counter_name, ranger.pred$predictions)
#tableAccuracy(drug.test$over_counter_name, ranger.pred$predictions)


###### Cross Validated RF with ranger package ######
set.seed(100)
#tuning parameters below. I'm not sure what about the splitrule and min.node.size, so I set them based off of the ranger documentation's recommendations 
tgrid <- expand.grid(
  .mtry = 1:10,
  .splitrule = "gini",
  .min.node.size = 1
)
#mtry should be number of variables (190)
#

drug.train.mm = as.data.frame(model.matrix(over_counter_name ~ condition + effectiveness + sideEffects + price + rating + usefulCount + sentiment,
                                           data = drug.train)) 
drug.test.mm = as.data.frame(model.matrix(over_counter_name ~ condition+ effectiveness + sideEffects + price + rating + usefulCount + sentiment, 
                                          data=drug.test))

# training on all except review
train.rf = train(over_counter_name ~ condition + effectiveness + sideEffects + price + rating + usefulCount + sentiment,
                 data = drug.train,
                 method = "ranger",
                 tuneGrid = tgrid,
                 trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE))
train.rf
mod.rf = train.rf$finalModel
pred.rf = predict(mod.rf, data = drug.test.mm)
table(drug.test$over_counter_name, pred.rf$predictions)
tableAccuracy(drug.test$over_counter_name, pred.rf$predictions)
### when trained on mtry 1:10, mtry = 10
# accuracy: 0.01806422


# training on review and conditions nlp only
train.rf2 = train(over_counter_name ~ .-effectiveness -sideEffects -price -rating -usefulCount -sentiment,
                 data = drug.train,
                 method = "ranger",
                 tuneGrid = tgrid,
                 trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE))
train.rf2

### when trained on mtry 1:10, mtry = 10
 
mod.rf2 = train.rf2$finalModel
pred.rf2 = predict(mod.rf2, data = drug.test.mm)
table(drug.test$over_counter_name, pred.rf2$predictions)
tableAccuracy(drug.test$over_counter_name, pred.rf2$predictions)

mod.rf = train.rf$finalModel
pred.rf = predict(mod.rf, data = drug.test)
table(drug.test$over_counter_name, pred.rf$predictions)
tableAccuracy(drug.test$over_counter_name, pred.rf$predictions)
# 0.3238156 accuracy with mtry = 10
######## predicting Over counter names without review #########

##### With Clustering ######
# pre-processing (normalize)
pp <- preProcess(drug, method=c("center", "scale"))
#step 2: apply it to our dataset
drug.scaled <- predict(pp, drug) 

drug.clusters <- kmeans(drug.scaled,297)

set.seed(144)
#km <- kmeans(airline.scaled, iter.max=100, 8)

