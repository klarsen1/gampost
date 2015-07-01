########## Clear the session
rm(list=ls())


########## Load packages
library(randomForestSRC)
library(e1071)
library(mgcv)
library(ggplot2)
library(grid)
library(kknn)
library(Information)

########## Setup
options(scipen=10)
ProjectLocation <- "/Users/kimlarsen/Google Drive/gampost/"

source(paste0(ProjectLocation, "/miscfunctions.R"))


########## Read the data
train <- readRDS(paste0(ProjectLocation, "/train.rda"))
valid <- readRDS(paste0(ProjectLocation, "/valid.rda"))

######### Kill the weakest variables with IV
IV <- Information::CreateTables(data=train, NULL, "PURCHASE", 10)
train <- train[,c(subset(IV$Summary, IV>0.05)$Variable, "PURCHASE")]
valid <- valid[,c(subset(IV$Summary, IV>0.05)$Variable, "PURCHASE")]

####### Check out a WOE table
View(IV$Tables$N_OPEN_REV_ACTS)

########## Random Forest
train$CPURCHASE <- as.factor(train$PURCHASE)
valid$CPURCHASE <- as.factor(valid$PURCHASE)

rf.init <- rfsrc(CPURCHASE ~ ., data=train[,!(names(train) == "PURCHASE")])
## Grow a forest to determine the number of trees and variables

imp <- cbind.data.frame(row.names(rf.init$importance), rf.init$importance[,"all"]) 
names(imp) <- c("Variable", "Importance")
imp <- imp[order(-imp$Importance),]
imp$Variable <- as.character(imp$Variable)
imp$Rank <- ave(1:nrow(imp), imp$Importance, FUN=mean)
View(imp)

plot(rf.init)
rm(rf.init)

system.time(
  rf.grow <- rfsrc(CPURCHASE ~ ., data=train[,c("CPURCHASE", subset(imp, Rank<21)$Variable)], ntree=100)
)
system.time(
  rf.pred <- predict(rf.grow, newdata=valid, outcome="train")
)
paste0("RF: ", AUC(valid$PURCHASE, rf.pred$predicted[,2])[1])
p <- plot.variable(x=rf.grow, "N_OPEN_REV_ACTS", partial=TRUE)
p_df <- cbind.data.frame(p$pData[[1]]$x.uniq, p$pData[[1]]$yhat)
names(p_df) <- c("x", "p_y")
p_df$p_y <- 1 - p_df$p_y
rfplot <- ggplot(data=p_df, aes(y=p_y, x=x)) + geom_line()

########## GAM using bam and variables selected by RandomForest, and smoothing parameters of 0.6.
f <- CreateGAMFormula(train[,subset(imp, Rank<21)$Variable], "PURCHASE", 0.6, "regspline")
system.time(
  gam1.model <- mgcv::gam(f, data=train, family=binomial(link="logit"))
)

### Plot a function using the lpmatrix:
x <- "N_OPEN_REV_ACTS"
gam1.lpmat <- predict(gam1.model, type="lpmatrix")
sxdf <- cbind.data.frame(train[[x]], gam1.lpmat[,grepl(x, colnames(gam1.lpmat))] %*% coef(gam1.model)[grepl(x, names(coef(gam1.model)))])
names(sxdf) <- c("x", "s_x")
sxdf$sx <- 1/(1+exp(-sxdf$s_x-coef(gam2.model)[1]))
gamplot <- ggplot(data=sxdf, aes(x=x, y=s_x)) + geom_line()


multiplot(rfplot, gamplot, cols=2)


### Predict the probabilities for the validation dataset.
system.time(
  gam1.predict <- 1/(1+exp(-predict(gam1.model, newdata=valid)))
)
paste0("GAM1: ", AUC(valid$PURCHASE, gam1.predict)[1])

########## GAM using bam and variables selected by RandomForest. Select smoothing parameters with REML.
f <- CreateGAMFormula(train[,subset(imp, Rank<21)$Variable], "PURCHASE", -1, "regspline")
system.time(
  gam2.model <- mgcv::gam(f, data=train, family=binomial(link="logit"), method="REML")
)

### Predict the probabilities for the validation dataset.
system.time(
  gam2.predict <- 1/(1+exp(-predict(gam2.model, newdata=valid)))
)
paste0("GAM2: ", AUC(valid[["PURCHASE"]], gam2.predict)[1])


########## SVM
## You can use tune.svm to tune the cost parameter, or fit the model directly 
## tuned <- tune(svm, CPURCHASE~., data=train[,!(names(train)=="PURCHASE")], cost=c(0.01, 0.1, 1), kernel="polynomial", degree=3, probability=TRUE)
## best model can be found in: tuned$best.model

system.time(
svm.model <- svm(CPURCHASE~., data=train[,subset(imp, Rank<21)$Variable], cost=0.01, kernel="polynomial", degree=3, probability=TRUE)
)

system.time(
svm.pred <- predict(svm.model,newdata=valid,probability=TRUE)
)
svm.prob <- as.numeric(attr(svm.pred, "probabilities")[,2])

paste0("SVM: ", AUC(valid[["PURCHASE"]], svm.prob)[1])


########## Logit model
logit.model <- glm(PURCHASE ~ ., data=train[,c("PURCHASE", subset(imp, Rank<21)$Variable)], family=binomial(link="logit"))
logit.predict <- 1/(1+exp(-predict(logit.model, newdata=valid)))
print(paste0("Linear logit: ", AUC(valid[["PURCHASE"]], logit.predict)[1]))

########## Logit model
logit.model <- glm(PURCHASE ~ ., data=train[,c("PURCHASE", subset(imp, Rank<21)$Variable)], family=binomial(link="logit"))
logit.predict <- 1/(1+exp(-predict(logit.model, newdata=valid)))
print(paste0("Linear logit: ", AUC(valid[["PURCHASE"]], logit.predict)[1]))

######### KNN classifier
system.time(
knn.classifier <- 
            kknn(CPURCHASE ~ ., 
            train=train[,c("CPURCHASE", subset(imp, Rank<21)$Variable)], 
            test=valid, 
            na.action = na.omit(),
            distance=2,
            k=100, 
            kernel = "epanechnikov", 
            scale=TRUE)
)

system.time(
paste0("KNN classifier: ", AUC(valid[["PURCHASE"]], knn.classifier$prob[,2])[1])
)

