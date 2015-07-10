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
ProjectLocation <- "/Users/kimlarsen/Documents/Code/gampost/"

source(paste0(ProjectLocation, "/miscfunctions.R"))

nvars <- 21


########## Read the data
train <- readRDS(paste0(ProjectLocation, "/train.rda"))
valid <- readRDS(paste0(ProjectLocation, "/valid.rda"))

######### Kill the weakest variables with IV
IV <- Information::CreateTables(data=train, NULL, "PURCHASE", 10)
View(IV$Summary)

train <- train[,c(subset(IV$Summary, IV>0.05)$Variable, "PURCHASE")]
valid <- valid[,c(subset(IV$Summary, IV>0.05)$Variable, "PURCHASE")]

####### Check out a WOE table
View(IV$Tables$N_OPEN_REV_ACTS)

########## Random Forest
train$CPURCHASE <- as.factor(train$PURCHASE)
valid$CPURCHASE <- as.factor(valid$PURCHASE)

## Grow an initial forest to determine the number of trees and variables
rf.init <- rfsrc(CPURCHASE ~ ., data=train[,!(names(train) == "PURCHASE")], ntree=500, seed=2015)

imp <- cbind.data.frame(row.names(rf.init$importance), rf.init$importance[,"1"]) 
names(imp) <- c("Variable", "Importance")
imp <- imp[order(-imp$Importance),]
imp$Variable <- as.character(imp$Variable)
imp$Rank <- ave(1:nrow(imp), imp$Importance, FUN=mean)
View(imp)

plot(rf.init)

system.time(
  rf.grow <- rfsrc(CPURCHASE ~ ., data=train[,c("CPURCHASE", subset(imp, Rank<nvars)$Variable)], ntree=100, seed=2015)
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

########## GAM using variables selected by RandomForest, and smoothing all parameters = 0.6.
f <- CreateGAMFormula(train[,subset(imp, Rank<nvars)$Variable], "PURCHASE", 0.6, "regspline")
system.time(
  gam1.model <- mgcv::gam(f, data=train, family=binomial(link="logit"))
)

### Plot a function using the lpmatrix:
x <- "N_OPEN_REV_ACTS"
gam1.lpmat <- predict(gam1.model, type="lpmatrix")
sxdf <- cbind.data.frame(train[[x]], gam1.lpmat[,grepl(x, colnames(gam1.lpmat))] %*% coef(gam1.model)[grepl(x, names(coef(gam1.model)))])
names(sxdf) <- c("x", "s_x")
sxdf$sx <- 1/(1+exp(-sxdf$s_x-coef(gam1.model)[1]))
gamplot <- ggplot(data=sxdf, aes(x=x, y=s_x)) + geom_line()


multiplot(rfplot, gamplot, cols=2)


### Predict the probabilities for the validation dataset.
system.time(
  gam1.predict <- 1/(1+exp(-predict(gam1.model, newdata=valid)))
)
paste0("GAM1: ", AUC(valid$PURCHASE, gam1.predict)[1])

########## GAM using bam and variables selected by RandomForest. Select smoothing parameters with REML.
f <- CreateGAMFormula(train[,subset(imp, Rank<nvars)$Variable], "PURCHASE", -1, "regspline")
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
svm.model <- svm(CPURCHASE~., data=train[,c("CPURCHASE", subset(imp, Rank<nvars)$Variable)], cost=0.001, gamma=0.000001, kernel="radial", degree=3, probability=TRUE)
)

system.time(
svm.pred <- predict(svm.model,newdata=valid,probability=TRUE)
)
svm.prob <- as.numeric(attr(svm.pred, "probabilities")[,2])

paste0("SVM: ", AUC(valid[["PURCHASE"]], svm.prob)[1])


########## Logit model
system.time(
logit.model <- glm(PURCHASE ~ ., data=train[,c("PURCHASE", subset(imp, Rank<nvars)$Variable)], family=binomial(link="logit"))
)
system.time(
logit.predict <- 1/(1+exp(-predict(logit.model, newdata=valid)))
)
print(paste0("Linear logit: ", AUC(valid[["PURCHASE"]], logit.predict)[1]))

######### KNN classifier
system.time(
knn.classifier <- 
            kknn(CPURCHASE ~ ., 
            train=train[,c("CPURCHASE", subset(imp, Rank<nvars)$Variable)], 
            test=valid, 
            na.action = na.omit(),
            distance=2,
            k=100, 
            kernel = "epanechnikov", 
            scale=TRUE)
)

paste0("KNN classifier: ", AUC(valid[["PURCHASE"]], knn.classifier$prob[,2])[1])

