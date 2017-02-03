install.packages("caret")
install.packages("tree")
install.packages("randomForest")
install.packages("e1071")
install.packages("nnet")
install.packages("rmarkdown")
install.packages("corrplot")

library(caret)
library(tree)
library(randomForest)
library(e1071)
library(nnet)
library(rmarkdown)
library(corrplot)


#import function from Github to plot nnet model
require(RCurl)

root.url<-'https://gist.githubusercontent.com/fawda123'
raw.fun<-paste(
  root.url,
  '5086859/raw/cc1544804d5027d82b70e74b83b3941cd2184354/nnet_plot_fun.r',
  sep='/'
)
script<-getURL(raw.fun, ssl.verifypeer = FALSE)
eval(parse(text = script))
rm('script','raw.fun')




set.seed(1022)
setwd("C:/Users/NicholasTan/Desktop/Studies/Gamma - Sem 2/TDS3301-Data Mining/Assignment/Part 3")
train_df <- read.csv("train.csv")
str(train_df)
summary(train_df)
head(train_df)

#Find out the correlation between the columns
corr <- cor(train_df[,1:94])
corrplot(corr, method="square", type="lower")


#Check for missing value
apply(train_df,2,function(x) sum(is.na(x)))


#Remove id because it is the same with the row number
train_df <- train_df[,-1]


#Preparing data for tree and ANN
train_size <- floor(0.75 * nrow(train_df))
train_ind <- sample(seq_len(nrow(train_df)), size = train_size)
train_tree <- train_df[train_ind, ]
test_tree <- train_df[-train_ind, ]

#Preparing data for naive bayes
train_df_factor <- as.data.frame(lapply(train_df, function(x) as.factor(x)))
train_ind_factor <- sample(seq_len(nrow(train_df_factor)), size = train_size)
train_NB <- train_df[train_ind_factor, ]
test_NB <- train_df[-train_ind_factor, ]

#Preparing data for ANN
train_ANN <- train_df[train_ind, ]
test_ANN <- train_df[-train_ind, ]



#Tree
tree_model <- tree(target ~ ., data=train_tree)
plot(tree_model)
title(main="tree")
text(tree_model)

#Test the model with predict
tree_pred <- predict(tree_model, test_tree, type="class")

#Create confusion matrix 
confusionMatrix(tree_pred, test_tree$target)
#Accuracy = 0.576

#Calculate the error rate for model
tree_model$error <- 1-(sum(tree_pred==test_tree$target)/length(test_tree$target))
tree_model$error
#Error rate = 0.4239819

#Prune the tree
tree_cv <- cv.tree(tree_model, FUN=prune.misclass)
names(tree_cv)
plot(tree_cv$size, tree_cv$dev, type="b")
plot(tree_cv$k, tree_cv$dev, type="b")
plot.tree.sequence(tree_cv)

pruned_model <- prune.misclass(tree_model, best=9)
plot(pruned_model)
text(pruned_model)

tree_pred <- predict(pruned_model, test_tree, type="class")
confusionMatrix(tree_pred, test_tree$target)

summary(pruned_model)

#Try use randomForest
#Calculating ideal value for mtry parameter
#mtry <- tuneRF(train_tree[,1:93], train_tree[,94], mtryStart=1,
#              ntreeTry=50, stepFactor=2, improve=0.05,
#              trace=TRUE, plot=TRUE, doBest=FALSE)

#Best value for mtry is 8

tree_model2 <- randomForest(train_tree$target ~ ., data=train_tree,
                            importance=TRUE,ntree=50,mtry=8)
#Create dotchart of variable/feature importance that is measured by random forest
varImpPlot(tree_model2)

tree_pred2 <- predict(tree_model2, test_tree, type="response")
confusionMatrix(tree_pred2, test_tree$target)

imp <- importance(tree_model2, type=1)
featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])
p <- ggplot(featureImportance, aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat="identity", fill="#53cfff") +
  coord_flip() + 
  theme_light(base_size=20) +
  xlab("Importance") +
  ylab("") + 
  ggtitle("Random Forest Feature Importance\n") +
  theme(plot.title=element_text(size=18))

ggsave("Feature_Importance.png", p, height=20, width=8, units="in")





#Naive Bayes
NB_model <- naiveBayes(train_NB$target ~ ., train_NB, laplace = 1)
NB_model

NB_pred <- predict(NB_model, test_NB[,-(ncol(test_NB))])
confusionMatrix(NB_pred, test_NB$target)





#ANN
NN_model <- nnet(train_ANN$target ~ ., train_ANN[,-1], size = 3, rang = 0.1, decay = 5e-4, maxit = 500)
plot.nnet(NN_model)
# Compute Predictions off Test Set
NN_pred <- predict(NN_model, test_ANN[,1:93],type="class")
confusionMatrix(test_ANN$target,NN_pred)