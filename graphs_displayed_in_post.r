library(mgcv)
library(demandr)
library(grid)
library(ggplot2)
library(splines)


### Simulated data
set.seed(3)
x <- seq(0,2*pi,0.1)
z <- sin(x)
y <- z + rnorm(mean=0, sd=0.5*sd(z), n=length(x))
d <- cbind.data.frame(x,y,z)

d1 <- cbind.data.frame(data.frame(predict(smooth.spline(x=d, spar=0), x)), z)
e <- mean((d1$z-d1$y)**2)
e
p1 <- ggplot(data=d, aes(x=x, y=y)) + geom_point() + geom_line(data=d1, aes(x=x, y=y), linetype=1) + geom_line(aes(x=x, y=z), linetype=2) + ggtitle(paste0("Lambda=0, MSE = ", round(e,2)))

d2 <- cbind.data.frame(data.frame(predict(smooth.spline(x=d, spar=0.3), x)), z)
e <- mean((d2$z-d2$y)**2)
e
p2 <- ggplot(data=d, aes(x=x, y=y)) + geom_point() + geom_line(data=d2, aes(x=x, y=y), linetype=1) + ylab("") + geom_line(aes(x=x, y=z), linetype=2) + ggtitle(paste0("Lambda=0.3, MSE = ", round(e,2)))

d3 <- cbind.data.frame(data.frame(predict(smooth.spline(x=d, spar=0.6), x)), z)
e <- mean((d3$z-d3$y)**2)
e
p3 <- ggplot(data=d, aes(x=x, y=y)) + geom_point() + geom_line(data=d3, aes(x=x, y=y), linetype=1) + ylab("") + geom_line(aes(x=x, y=z), linetype=2) + ggtitle(paste0("Lambda=0.6, MSE = ", round(e,2)))

d4 <- cbind.data.frame(data.frame(predict(smooth.spline(x=d, spar=1), x)), z)
e <- mean((d4$z-d4$y)**2)
e
p4 <- ggplot(data=d, aes(x=x, y=y)) + geom_point() + geom_line(data=d4, aes(x=x, y=y), linetype=1) + ylab("") + geom_line(aes(x=x, y=z), linetype=2) + ggtitle(paste0("Lambda=1, MSE = ", round(e,2)))

multiplot(p1, p2, p3, p4, cols=2)

d5 <- cbind.data.frame(data.frame(ksmooth(d$x, d$y, kernel="box", n.points=length(x), bandwidth=1.5)), z)
e <- mean((d5$z-d5$y)**2)
e
p5 <- ggplot(data=d, aes(x=x, y=y)) + geom_point() + geom_line(data=d5, aes(x=x, y=y), linetype=1) + ylab("") + geom_line(aes(x=x, y=z), linetype=2) + ggtitle(paste0("Basic Runnuing Mean, MSE = ", round(e,2)))

d6 <- cbind.data.frame(loess(y ~ x, data=d, span=0.6)$fitted, z, y, x)
names(d6) <- c("loess", "z", "y", "x")
#d6 <- cbind.data.frame(data.frame(ksmooth(d$x, d$y, kernel="normal", n.points=length(x), bandwidth=1)), z)
e <- mean((d6$z-d6$y)**2)
e
p6 <- ggplot(data=d, aes(x=x, y=y)) + geom_point() + geom_line(data=d6, aes(x=x, y=loess), linetype=1) + ylab("") + geom_line(aes(x=x, y=z), linetype=2) + ggtitle(paste0("Loess, MSE = ", round(e,2)))

multiplot(p5, p6, cols=2)

min(x)

max(x)

quantile(x, probs=c(0.25, .50, .75))


B <- bs(x, degree=3, intercept=TRUE, Boundary.knots=c(0, 6.2), knots=c(1.55, 3.10, 4.65))
model <- lm(y~0 + B)

model$coef
d7 <- cbind.data.frame(d, B, model$fitted)
names(d7) <- c("x", "y", "z", "B13", "B23", "B33", "B43", "B53", "B63", "B73", "Spline")
for (i in 1:7){
   d7[,3+i] <- d7[,3+i] * model$coef[i]
}

e <- mean((d7$z-d7$Spline)**2)
e
ggplot(data=d7, aes(x=x, y=y)) + geom_point() + geom_line(data=d7, aes(x=x, y=Spline), linetype=1) + ylab("") +  geom_line(aes(x=x, y=z), linetype=2)
p7 <- ggplot(data=d7, aes(x=x, y=y)) + geom_point() + geom_line(data=d7, aes(x=x, y=Spline), linetype=1) + ylab("") +  geom_line(aes(x=x, y=z), linetype=2) + ggtitle(paste0("B-Spline, MSE = ", round(e,2)))
p7

d7_melt <- melt(d7[,c("x", "B13", "B23", "B33", "B43", "B53", "B63", "B73", "Spline")], id.vars="x")

line.cols <- terrain.colors(8)
line.cols[8] <- "black"
ggplot(data=d7_melt, aes(y=value, x=x, colour=variable)) + geom_line() + scale_color_manual(values=line.cols) + ylab("")  

        

