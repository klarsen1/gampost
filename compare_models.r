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
library(ClustOfVar)
library(reshape2)
library(plyr)

########## Setup
options(scipen=10)
ProjectLocation <- "/Users/kimlarsen/Documents/Code/gampost/"

source(paste0(ProjectLocation, "/miscfunctions.R"))

nvars <- 20


########## Read the data
train <- readRDS(paste0(ProjectLocation, "/train.rda"))
valid <- readRDS(paste0(ProjectLocation, "/valid.rda"))

######### Kill the weakest variables with IV
IV <- Information::CreateTables(data=train, NULL, "PURCHASE", 10)
View(IV$Summary)

train <- train[,c(subset(IV$Summary, IV>0.05)$Variable, "PURCHASE")]
valid <- valid[,c(subset(IV$Summary, IV>0.05)$Variable, "PURCHASE")]

######## Variable clustering
tree <- hclustvar(train[,!(names(train)=="PURCHASE")])
##nclusters <- length(tree[tree$height<0.7])
part_init<-cutreevar(tree,nvars)$cluster
kmeans<-kmeansvar(X.quanti=train[,!(names(train)=="PURCHASE")],init=part_init)
clusters <- cbind.data.frame(melt(kmeans$cluster), row.names(melt(kmeans$cluster)))
names(clusters) <- c("Cluster", "Variable")
clusters <- join(clusters, IV$Summary, by="Variable", type="left")
clusters <- clusters[order(clusters$Cluster),]
clusters$Rank <- ave(-clusters$IV, clusters$Cluster, FUN=rank)
View(clusters)
variables <- as.character(subset(clusters, Rank==1)$Variable)


####### Check out a WOE table
View(IV$Tables$N_OPEN_REV_ACTS)
View(IV$Tables$RATIO_RETAIL_BAL2HI_CRDT)

########## Random Forest
train$CPURCHASE <- ifelse(train$PURCHASE==1, 1, -1)
valid$CPURCHASE <- ifelse(valid$PURCHASE==1, 1, -1)
train$CPURCHASE <- as.factor(train$CPURCHASE)
valid$CPURCHASE <- as.factor(valid$CPURCHASE)

system.time(
  rf.grow <- rfsrc(CPURCHASE ~ ., data=train[,c("CPURCHASE", variables)], ntree=100, seed=2015)
)
system.time(
  rf.pred <- predict(rf.grow, newdata=valid, outcome="train")
)
paste0("RF: ", AUC(valid$PURCHASE, rf.pred$predicted[,2])[1])

rf.grow$importance

p <- plot.variable(x=rf.grow, "N_OPEN_REV_ACTS", partial=TRUE)
p_df <- cbind.data.frame(p$pData[[1]]$x.uniq, p$pData[[1]]$yhat)
names(p_df) <- c("x", "p_y")
p_df$p_y <- 1 - p_df$p_y
rfplot1 <- ggplot(data=p_df, aes(y=p_y, x=x)) + geom_line() + ggtitle("Random Forest") + scale_y_continuous(limits = c(0.15, 0.6)) + ylab("P(Y=1)")

p <- plot.variable(x=rf.grow, "RATIO_RETAIL_BAL2HI_CRDT", partial=TRUE)
p_df <- cbind.data.frame(p$pData[[1]]$x.uniq, p$pData[[1]]$yhat)
names(p_df) <- c("x", "p_y")
p_df$p_y <- 1 - p_df$p_y
rfplot2 <- ggplot(data=subset(p_df, x<=100), aes(y=p_y, x=x)) + geom_line() + ggtitle("Random Forest") + scale_y_continuous(limits = c(0.15, 0.3)) + ylab("P(Y=1)")


########## GAM using variables selected by RandomForest, and smoothing all parameters = 0.6.
f <- CreateGAMFormula(train[,variables], "PURCHASE", 0.6, "regspline")
system.time(
  gam1.model <- mgcv::gam(f, data=train, family=binomial(link="logit"))
)

## Check the concurvity
#concurvity(gam1.model,full=FALSE)

### Predict the probabilities for the validation dataset.
system.time(
  gam1.predict <- 1/(1+exp(-predict(gam1.model, newdata=valid)))
)
paste0("GAM1: ", AUC(valid$PURCHASE, gam1.predict)[1])

########## GAM where smoothing parameters are selected with REML.
f <- CreateGAMFormula(train[,variables], "PURCHASE", -1, "regspline")
system.time(
  gam2.model <- mgcv::gam(f, data=train, family=binomial(link="logit"), method="REML")
)

### Predict the probabilities for the validation dataset.
system.time(
  gam2.predict <- 1/(1+exp(-predict(gam2.model, newdata=valid)))
)
paste0("GAM2: ", AUC(valid[["PURCHASE"]], gam2.predict)[1])

########## GAM where smoothing parameters are selected with REML and weak variables are shrunk (selection=TRUE).
f <- CreateGAMFormula(data=train[,variables], y="PURCHASE", type="none")
system.time(
  gam3.model <- mgcv::gam(f, data=train, family=binomial(link="logit"), method="REML", select=TRUE)
)

#, control = list(nthreads=4) 
### Predict the probabilities for the validation dataset.
system.time(
  gam3.predict <- 1/(1+exp(-predict(gam3.model, newdata=valid)))
)
paste0("GAM3: ", AUC(valid[["PURCHASE"]], gam3.predict)[1])

### Plot a function using the lpmatrix:
x <- "N_OPEN_REV_ACTS"
gam1.lpmat <- predict(gam1.model, type="lpmatrix")
sxdf <- cbind.data.frame(train[[x]], gam1.lpmat[,grepl(x, colnames(gam1.lpmat))] %*% coef(gam1.model)[grepl(x, names(coef(gam1.model)))])
names(sxdf) <- c("x", "s_x")
sxdf$sx <- 1/(1+exp(-sxdf$s_x-coef(gam1.model)[1]))
gamplot1 <- ggplot(data=sxdf, aes(x=x, y=sx)) + geom_line() + ggtitle("GAM (lambda=0.6)") + scale_y_continuous(limits = c(0.15, 0.6))  + ylab("")

x <- "RATIO_RETAIL_BAL2HI_CRDT"
gam1.lpmat <- predict(gam1.model, type="lpmatrix")
sxdf <- cbind.data.frame(train[[x]], gam1.lpmat[,grepl(x, colnames(gam1.lpmat))] %*% coef(gam1.model)[grepl(x, names(coef(gam1.model)))])
names(sxdf) <- c("x", "s_x")
sxdf$sx <- 1/(1+exp(-sxdf$s_x-coef(gam1.model)[1]))
gamplot2 <- ggplot(data=subset(sxdf, x<=100), aes(x=x, y=sx)) + geom_line() + ggtitle("GAM (lambda=0.6)") + scale_y_continuous(limits = c(0.15, 0.30)) +ylab("")

multiplot(rfplot1, gamplot1, cols=2)

multiplot(rfplot2, gamplot2, cols=2)

########## SVM (radial Gaussian kernel)
## You can use tune.svm to tune the cost parameter, or fit the model directly 
## tuned <- tune(svm, CPURCHASE~., data=train[,!(names(train)=="PURCHASE")], cost=c(0.01, 0.1, 1), kernel="polynomial", degree=3, probability=TRUE)
## best model can be found in: tuned$best.model

system.time(
svm.model1 <- svm(CPURCHASE~., data=train[,c("CPURCHASE", variables)], cost=0.001, gamma=0.000001, kernel="radial", probability=TRUE)
)

system.time(
svm.pred1 <- predict(svm.model1,newdata=valid,probability=TRUE)
)
svm.prob1 <- as.numeric(attr(svm.pred1, "probabilities")[,2])

paste0("SVM (radial): ", AUC(valid[["PURCHASE"]], svm.prob1)[1])

########## SVM (poly kernel, order=3)
## You can use tune.svm to tune the cost parameter, or fit the model directly 
## tuned <- tune(svm, CPURCHASE~., data=train[,!(names(train)=="PURCHASE")], cost=c(0.01, 0.1, 1), kernel="polynomial", degree=3, probability=TRUE)
## best model can be found in: tuned$best.model

system.time(
  svm.model2 <- svm(CPURCHASE~., data=train[,c("CPURCHASE", variables)], cost=0.001, kernel="polynomial", degree=3, probability=TRUE)
)

system.time(
  svm.pred2 <- predict(svm.model2,newdata=valid,probability=TRUE)
)
svm.prob2 <- as.numeric(attr(svm.pred2, "probabilities")[,2])

print(attr(svm.pred2))

paste0("SVM (poly): ", AUC(valid[["PURCHASE"]], svm.prob2)[1])

########## Logit model
system.time(
logit.model <- glm(PURCHASE ~ ., data=train[,c("PURCHASE", variables)], family=binomial(link="logit"))
)
system.time(
logit.predict <- 1/(1+exp(-predict(logit.model, newdata=valid)))
)
print(paste0("Linear logit: ", AUC(valid[["PURCHASE"]], logit.predict)[1]))

######### KNN classifier
system.time(
knn.classifier <- 
            kknn(CPURCHASE ~ ., 
            train=train[,c("CPURCHASE", variables)], 
            test=valid, 
            na.action = na.omit(),
            distance=2,
            k=100, 
            kernel = "epanechnikov", 
            scale=TRUE)
)

paste0("KNN classifier: ", AUC(valid[["PURCHASE"]], knn.classifier$prob[,2])[1])

