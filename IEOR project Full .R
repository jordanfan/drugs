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

# this data set uses detailed pricing (to the 4th decimal place)
drug = read.csv("drugs_final1.csv", stringsAsFactors=FALSE, na.strings = c("", "NA"))
drug$drugName <- NULL

####### removing rows where over_counter_name has frequency <=50 #####
sorted_counts = as.data.frame(sort(table(drug$over_counter_name)))
keep_drugs = sorted_counts$Var1[106:nrow(sorted_counts)] 
drug = drug[drug$over_counter_name %in% keep_drugs,]

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

#################     Default Ranger - NLP (review and conditions)   #####################
ranger.nlp.default = ranger(over_counter_name ~ .-effectiveness -sideEffects -price -rating -usefulCount -sentiment,
                            data = drug.train, verbose = TRUE)

pred.rg.nlp.d = predict(ranger.nlp.default, data = drug.test)
tableAccuracy(drug.test$over_counter_name, pred.rg.nlp.d$predictions)
# accuracy = 0.3122618

###############   Cross Validated NLP (features: review and condition)  ####################
#took 2.5 hours
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

#################     mtry = 10 Ranger - NLP (review and conditions)   #####################
#took a couple minutes
ranger.nlp = ranger(over_counter_name ~ .-effectiveness -sideEffects -price -rating -usefulCount -sentiment,
                    data = drug.train, mtry = 10, verbose = TRUE)

pred.rg.nlp = predict(ranger.nlp, data = drug.test)
tableAccuracy(drug.test$over_counter_name, pred.rg.nlp$predictions)
# accuracy = 0.3134932


################   Default Ranger - All features   ######################

ranger.all.default = ranger(over_counter_name ~ .,
                            data = drug.train, verbose = TRUE)

pred.rg.all.d = predict(ranger.all.default, data = drug.test)
tableAccuracy(drug.test$over_counter_name, pred.rg.all.d$predictions)
#accuracy = 0.7733537

set.seed(100)
tgrid <- expand.grid(
  .mtry = 1:130,
  .splitrule = "gini",
  .min.node.size = 1
)

################     CV using train() - All features   ##################

train.rf.all = train(over_counter_name ~ .,
                     data = drug.train,
                     method = "ranger",
                     tuneGrid = tgrid,
                     trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE))

# Testing mtry 1:130
# mtry = 124

mod.rf.all = train.rf.all$finalModel
pred.rf.all = predict(mod.rf.all, data = drug.test.mm.all)
tableAccuracy(drug.test$over_counter_name, pred.rf.all$predictions)
# Accuracy 0.004867179

##############  setting Ranger to mtry based off of CV - All features    ##################
#takes like 2-3 minutes
ranger.all = ranger(over_counter_name ~ .,
                    data = drug.train, mtry = 124, verbose = TRUE)

pred.rg.all = predict(ranger.all, data = drug.test)
tableAccuracy(drug.test$over_counter_name, pred.rg.all$predictions)
# accuracy = 0.9998241
#  #  #

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################


#### This dataset contains the binned prices ###

drug = read.csv("Final_Data.csv", stringsAsFactors=FALSE, na.strings = c("", "NA"))
drug$drugName <- NULL

####### removing rows where over_counter_name has frequency <=50 #####
sorted_counts = as.data.frame(sort(table(drug$over_counter_name)))
keep_drugs = sorted_counts$Var1[106:nrow(sorted_counts)] 
drug = drug[drug$over_counter_name %in% keep_drugs,]

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
drugData$price = as.factor(drug$price)
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

###################     baseline model     ########################
#baseline predicts citalopram (911 instances in training data)
# in test data, citalopram appears 390 times
#accuracy: 390 / 16990 = 0.02295468

table(drug.train$over_counter_name)
x <-count(drug.train,over_counter_name)
summary(x) # to find max count
table(drug.test$over_counter_name)



drug.train.mm.nlp = as.data.frame(model.matrix(over_counter_name ~ . -1 -effectiveness -sideEffects -price -rating -usefulCount -sentiment, 
                                         data=drug.train))
drug.train.mm.all = as.data.frame(model.matrix(over_counter_name ~ . -1, 
                                               data=drug.train))
drug.test.mm.nlp = as.data.frame(model.matrix(over_counter_name ~ . -1 -effectiveness -sideEffects -price -rating -usefulCount -sentiment, 
                                          data=drug.test))
drug.test.mm.all = as.data.frame(model.matrix(over_counter_name ~ . -1, 
                                          data=drug.test))

set.seed(100)
tgrid <- expand.grid(
  .mtry = 1:130,
  .splitrule = "gini",
  .min.node.size = 1
)
#################     Default Ranger - NLP (review and conditions)    #####################
ranger.nlp.default = ranger(over_counter_name ~ .-effectiveness -sideEffects -price -rating -usefulCount -sentiment,
                            data = drug.train, verbose = TRUE)

pred.rg.nlp.d = predict(ranger.nlp.default, data = drug.test)
tableAccuracy(drug.test$over_counter_name, pred.rg.nlp.d$predictions)
# accuracy = 0.3130665

###################    Cross Validated NLP (review and conditions)    ####################
#took 2.5 hours
train.rf.nlp = train(over_counter_name ~ . -effectiveness -sideEffects -price -rating -usefulCount -sentiment,
                     data = drug.train,
                     method = "ranger",
                     tuneGrid = tgrid,
                     trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE))

# Testing mtry 1:20
# (sqrt(p)) is around 11-12
# CV mtry = 11

mod.rf.nlp = train.rf.nlp$finalModel 
pred.rf.nlp = predict(mod.rf.nlp, data = drug.test.mm.nlp)
table(drug.test$over_counter_name, pred.rf.nlp$predictions)
tableAccuracy(drug.test$over_counter_name, pred.rf.nlp$predictions)
# accuracy =  0.3138317

#################     mtry = 11 Ranger - NLP (review and conditions)    #####################
#took a couple minutes
ranger.nlp = ranger(over_counter_name ~ .-effectiveness -sideEffects -price -rating -usefulCount -sentiment,
                    data = drug.train, mtry = 11, verbose = TRUE)

pred.rg.nlp = predict(ranger.nlp, data = drug.test)
tableAccuracy(drug.test$over_counter_name, pred.rg.nlp$predictions)
# accuracy =  0.3130665



################   Default Ranger - All features   ######################

ranger.all.default = ranger(over_counter_name ~ .,
                            data = drug.train, verbose = TRUE)

pred.rg.all.d = predict(ranger.all.default, data = drug.test)
tableAccuracy(drug.test$over_counter_name, pred.rg.all.d$predictions)
#accuracy = 0.5671572

###################     CV using train()- All features   ########################
#took 30 hours
train.rf.all = train(over_counter_name ~ .,
                  data = drug.train,
                  method = "ranger",
                  tuneGrid = tgrid,
                  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE))

# Testing mtry 1:130
#mtry = 25

mod.rf.all = train.rf.all$finalModel
pred.rf.all = predict(mod.rf.all, data = drug.test.mm.all)
tableAccuracy(drug.test$over_counter_name, pred.rf.all$predictions)
# accuracy = 0.2301942

ggplot(train.rf.all$results, aes(x = mtry, y = Accuracy)) + geom_point(size = 2) + geom_line() + 
  ylab("CV Accuracy") + theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18)) + ggtitle("Random Forest CV on all Features")

################   Ranger set to the CV mtry value - All features   ######################
#takes like 2-3 minutes
ranger.all = ranger(over_counter_name ~ .,
                 data = drug.train, mtry = 25, verbose = TRUE)

pred.rg.all = predict(ranger.all, data = drug.test)
tableAccuracy(drug.test$over_counter_name, pred.rg.all$predictions)
# accuracy = 0.581342
#  #  #

#################   CV CART - ALL features  ######################
cpVals = data.frame(cp = seq(0, .1, by=.005))
train.cart <- train(over_counter_name ~ .,
                    data = drug.train,
                    method = "rpart",
                    tuneGrid = cpVals,
                    trControl = trainControl(method = "cv", number=5),
                    metric = "Accuracy")

train.cart$results 
train.cart
ggplot(train.cart$results, aes(x=cp, y=Accuracy)) + geom_point()+ggtitle("Cart CP")
train.cart$bestTune
#cp = 0
mod123 = train.cart$finalModel
prp(mod123)
pred.cart = predict(mod123, newdata=drug.test.mm.all, type="class")
tableAccuracy(drug.test$over_counter_name, pred.cart)
#accuracy = 0.4864626

###############################################################################

##################### Bootstraps ############################

library(boot)

mean_squared_error <- function(data, index) {
  responses <- data$response[index]
  predictions <- data$prediction[index]
  MSE <- mean((responses - predictions)^2)
  return(MSE)
}
mean_absolute_error <- function(data, index) {
  responses <- data$response[index]
  predictions <- data$prediction[index]
  MAE <- mean(abs(responses - predictions))
  return(MAE)
}
OS_R_squared <- function(data, index) {
  responses <- data$response[index]
  predictions <- data$prediction[index]
  baseline <- data$baseline[index]
  SSE <- sum((responses - predictions)^2)
  SST <- sum((responses - baseline)^2)
  r2 <- 1 - SSE/SST
  return(r2)
}
OSR2 <- function(prediction, test, train) {
  SSE = sum((test - prediction)^2)
  SST = sum((test - mean(train))^2)
  OSR2 = 1 - SSE/SST
  return (OSR2)
}

accuracy <- function(data, index) {
  responses <- data$response[index]
  predictions <- data$prediction[index]
  
  acc <- tableAccuracy(responses, predictions)
  return (acc)
  
}
all_metrics <- function(data, index) {
  mse <- mean_squared_error(data, index)
  mae <- mean_absolute_error(data, index)
  OSR2 <- OS_R_squared(data, index)
  accurate <- accuracy(data, index)
  return(c(mse, mae, OSR2, accurate))
}

####### bootstrap for train.rf.nlp (cross validated rf model on review and condition) ###########
nlp_rf_test_set = data.frame(response = drug.test$over_counter_name, prediction = pred.rf.nlp$predictions)
# do bootstrap
set.seed(892)
nlp_rf_boot <- boot(nlp_rf_test_set, all_metrics, R = 10000)
nlp_rf_boot
#Bootstrap Statistics :
#original        bias    std. error
#t4* 0.3138317 -7.208946e-05 0.003603598

boot.ci(nlp_rf_boot, index = 4, type = "perc")
#Intervals : 
#  Level     Percentile     
#95%   ( 0.3068,  0.3208 )

# make plots
nlp_rf_boot_plot_results = data.frame(accuracyestimates = nlp_rf_boot$t[,4], delta = nlp_rf_boot$t[,4] - nlp_rf_boot$t0[4])

quantile(nlp_rf_boot_plot_results$accuracyestimates, c(0.025, 0.975))
#      2.5%     97.5% 
#0.3067687 0.3208358 

ggplot(nlp_rf_boot_plot_results) + geom_histogram(aes(x = accuracyestimates), binwidth = 0.005, color = "blue") + 
  ylab("Count") + xlab("Bootstrap Accuracy Estimate") + theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18))+ 
  geom_vline(xintercept = 0.3067687 ) +
  geom_vline(xintercept = 0.3208358 ) + 
  ggtitle("Bootstrap CI for CV RF model on Review and Conditions")


####### bootstrap for train.rf.all (cross validated rf model on all features) ###########
all_rf_test_set = data.frame(response = drug.test$over_counter_name, prediction = pred.rf.all$predictions)
# do bootstrap
set.seed(892)
all_rf_boot <- boot(all_rf_test_set, all_metrics, R = 10000)
all_rf_boot
#Bootstrap Statistics :
#original       bias    std. error
#t4* 0.2301942 -4.06239e-05 0.003218936

boot.ci(all_rf_boot, index = 4, type = "perc")
#Intervals : 
#Level     Percentile     
#95%   ( 0.2238,  0.2364 )  

# make plots
all_rf_boot_plot_results = data.frame(accuracyestimates = all_rf_boot$t[,4], delta = all_rf_boot$t[,4] - all_rf_boot$t0[4])

quantile(all_rf_boot_plot_results$accuracyestimates, c(0.025, 0.975))
#    2.5%     97.5% 
#0.2238376 0.2363170     

ggplot(all_rf_boot_plot_results) + geom_histogram(aes(x = accuracyestimates), binwidth = 0.005, color = "blue") + 
  ylab("Count") + xlab("Bootstrap Accuracy Estimate") + theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18))+ 
  geom_vline(xintercept = 0.2238376 ) +
  geom_vline(xintercept = 0.2363170 ) + 
  ggtitle("Bootstrap CI for CV RF model on All Features")



####### bootstrap for ranger.all (Ranger model built on all features and mtry = 25) ###########
all_rg_test_set = data.frame(response = drug.test$over_counter_name, prediction = pred.rg.all$predictions)
# do bootstrap
set.seed(892)
all_rg_boot <- boot(all_rg_test_set, all_metrics, R = 10000)
all_rg_boot
#Bootstrap Statistics :
#original       bias    std. error
#t4* 0.581342 1.982931e-05 0.003767488

boot.ci(all_rg_boot, index = 4, type = "perc")
#Intervals : 
#Level     Percentile     
#95%   ( 0.5741,  0.5887 ) 

# make plots
all_rg_boot_plot_results = data.frame(accuracyestimates = all_rg_boot$t[,4], delta = all_rg_boot$t[,4] - all_rg_boot$t0[4])

quantile(all_rg_boot_plot_results$accuracyestimates, c(0.025, 0.975))
#      2.5%     97.5% 
#0.5741024 0.5886992      

ggplot(all_rg_boot_plot_results) + geom_histogram(aes(x = accuracyestimates), binwidth = 0.005, color = "blue") + 
  ylab("Count") + xlab("Bootstrap Accuracy Estimate") + theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18))+ 
  geom_vline(xintercept = 0.5741024 ) +
  geom_vline(xintercept = 0.5886992 ) + 
  ggtitle("Bootstrap CI for RF model on All Features, using cross-validated mtry")


#### bootstrap for CART #####
cart_test_set = data.frame(response = drug.test$over_counter_name, prediction = pred.cart)
# do bootstrap
set.seed(892)
cart_boot <- boot(cart_test_set, all_metrics, R = 10000)
cart_boot
#Bootstrap Statistics :
#original       bias    std. error
#t4* 0.4864626 0.0001018128 0.003796544

boot.ci(cart_boot, index = 4, type = "perc")
#Intervals : 
#Level     Percentile     
#95%   ( 0.4790,  0.4939 )  

# make plots
cart_boot_plot_results = data.frame(accuracyestimates = cart_boot$t[,4], delta = cart_boot$t[,4] - cart_boot$t0[4])
quantile(cart_boot_plot_results$accuracyestimates, c(0.025, 0.975))
#  2.5%     97.5% 
#  0.4791039 0.4939376 
ggplot(cart_boot_plot_results) + geom_histogram(aes(x = accuracyestimates), binwidth = 0.005, color = "blue") + 
  ylab("Count") + xlab("Bootstrap Accuracy Estimate") + theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18))+ 
  geom_vline(xintercept = 0.4791039) +
  geom_vline(xintercept = 0.4939376 ) + 
  ggtitle("Bootstrap CI for CART")



#### bootstrap for ranger.all.default (default rf on all variables) ###
all_default_rf_set = data.frame(response = drug.test$over_counter_name, prediction = pred.rg.all.d$predictions, baseline = 0)
# do bootstrap
set.seed(892)
all_default_rf_boot <- boot(all_default_rf_set, all_metrics, R = 10000)
all_default_rf_boot
#Bootstrap Statistics :
#  original       bias    std. error
#t4* 0.5671572 4.546792e-05 0.003815011

boot.ci(all_default_rf_boot, index = 4, type = "perc")
#Intervals : 
#Level     Percentile     
#95%   ( 0.5597,  0.5746 ) 

all_default_rf_boot_results = data.frame(accuracyestimates = all_default_rf_boot$t[,4], delta = all_default_rf_boot$t[,4] - all_default_rf_boot$t0[4])
quantile(all_default_rf_boot_results$accuracyestimates, c(0.025, 0.975))
#   2.5%     97.5% 
#0.5597396 0.5745733 
ggplot(all_default_rf_boot_results) + geom_histogram(aes(x = accuracyestimates), binwidth = 0.005, color = "blue") + 
  ylab("Count") + xlab("Bootstrap Accuracy Estimate") + theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18))+ 
  geom_vline(xintercept = 0.5597396) +
  geom_vline(xintercept = 0.5745733 ) + 
  ggtitle("Bootstrap CI for default RF on All Variables")


















