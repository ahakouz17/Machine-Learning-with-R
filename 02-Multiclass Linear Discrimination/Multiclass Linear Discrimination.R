##################################################################################
###       COMP/INDR 421/521 INTRODUCTION TO MACHINE LEARNING (Fall 2017)       ###
###                HW02: Multiclass Linear Discrimination                      ###      
###                Developed by: Asma Hakouz                                   ### 
##################################################################################
library(MASS)
## GENERATING DATA
# choose an arbitrary value for the seed which will be used for random numbers generation
set.seed(401)

# define classes parameters, each with 2 inputs(features); x1 & x2
# mean parameters
class1_means <- c(0.0, 1.5)
class2_means <- c(-2.5, -3.0)
class3_means <- c(2.5, -3.0)

# Coveriance matrices
class1_sigma <- matrix(c(1, 0.2, 0.2, 3.2), 2, 2)
class2_sigma <- matrix(c(1.6, -0.8, -0.8, 1.0), 2, 2)
class3_sigma <- matrix(c(1.6, 0.8, 0.8, 1.0), 2, 2)

# sample sizes
class_sizes <- c(100, 100, 100)

# generate random samples from multivariate (in our case it's bivariate) normal distributions
points1 <- MASS::mvrnorm(n = class_sizes[1], class1_means, class1_sigma)
points2 <- MASS::mvrnorm(n = class_sizes[2], class2_means, class2_sigma)
points3 <- MASS::mvrnorm(n = class_sizes[3], class3_means, class3_sigma)

# plot the generated data points from all classes.  
plot(points1[,1], points1[,2], type = "p", col = rgb(0.2,0.4,0.1,0.9), lwd = 0.5, 
     xlab = "x1", ylab = "x2", ylim = c(-6, max(points1[,2], points2[,2], points3[,2])),
     xlim = c(-6, max(points1[,1], points2[,1], points3[,1])), pch = 19)
points(points2[,1], points2[,2], type = "p", col = rgb(0.8,0.4,0.1,0.9), lwd = 0.5, pch = 15)
points(points3[,1], points3[,2], type = "p", col = rgb(0.1,0.5,0.4,0.9), lwd = 0.5, pch = 17)

legend("topleft", 
       legend = c("class1", "class2", "class3"), 
       col = c(rgb(0.2,0.4,0.1,0.9), 
               rgb(0.8,0.4,0.1,0.9),
               rgb(0.1,0.5,0.4,0.9)), 
       pch = c(19, 15, 17), bty = "n", pt.cex = 1.5, cex = 1.2, 
       text.col = rgb(0.3,0.3,0.3,1), horiz = F, inset = c(0.01, 0.01))


x1 <- c(points1[,1], points2[,1], points3[,1])
x2 <- c(points1[,2], points2[,2], points3[,2])

# generate corresponding labels
y <- c(rep(1, class_sizes[1]), rep(2, class_sizes[2]), rep(3, class_sizes[3]))

# write data to a file
write.csv(x = cbind(x1, x2, y), file = "HW02_data_set.csv", row.names = FALSE)

##################################################################################
## DATA PROCESSING AND ALGORITHM
# read data into memory
data_set <- read.csv("HW02_data_set.csv")

# get x1, x2 and y values
x1 <- data_set$x1
x2 <- data_set$x2
X <- cbind(x1, x2)
y_truth <- data_set$y

# get number of classes(K) number of samples(N), 
# and number of inputs/features(d)
K <- max(y_truth)
N <- length(y_truth)
d <- ncol(X)

# set learning parameters
eta <- 0.02
epsilon <- 1e-3

# randomly initalize w and w0
set.seed(401)
w <- matrix(runif(ncol(X)*K, min = -0.01, max = 0.01), d, K)
w0 <- matrix(runif(K, min = -0.01, max = 0.01), 1, K)

# learn w and w0 using gradient descent
iteration <- 1
objective_values <- c()

# define the gradient functions
gradient_w <- function(X, y_predicted_val, y_truth) {
  delta_w <- c()
  for(c in 1:3){
    sum <- 0
    for(s in 1:300){
      if(y_truth[s] == c){
        sum <- sum + (1 - y_predicted_val[s, c]) * X[s,]
      }
      else {
        sum <- sum - y_predicted_val[s, c] * X[s,]
      }
    }
    delta_w <- cbind(delta_w, sum)
  }
  return (delta_w)
}

gradient_w0 <- function(y_predicted_val, y_truth) {
  delta_w0 <- c()
  for(c in 1:3){
    sum <- 0
    for(s in 1:300){
      if(y_truth[s] == c){
        sum <- sum + (1 - y_predicted_val[s, c])
      } else {
        sum <- sum - y_predicted_val[s, c]
      }
    }
    delta_w0 <- cbind(delta_w0, sum)
  }
  return (delta_w0)
}

# define the softmax function
softmax <- function(X, w, w0, k, c){
  denomSum <- 0
  for(i in 1:k){
    denomSum <- denomSum + exp(X %*% w[,i] + w0[i])
  }
  return (exp(X %*% w[,c] + w0[c])/denomSum)
}

# define class predicting function
predictClass <- function(X, w, w0, k, N) {
  softmaxScore <- c()
  # calculate score functions for all classes and choose the one with the 
  # highest score / probability using softmax function
  for(i in 1:k){
    softmaxScore <- cbind(softmaxScore, softmax(X, w, w0, k, i))
  }
  #print(softmaxScore)
  y_predicted <- c()
  y_predicted_probability <- c()
  y_predicted_val <- c()
  for(j in 1:N){
    maxScore <- -1
    mostProbableClass <- -1
    for(i in 1:k){
      if(softmaxScore[j,i] > maxScore){
        maxScore <- softmaxScore[j,i]
        mostProbableClass <- i
      }
    }
    y_predicted <- rbind(y_predicted, mostProbableClass)
    y_predicted_val <- rbind(y_predicted_val, maxScore)
  }
  return(cbind(y_predicted, softmaxScore, y_predicted_val))
}

while (1) {
  print(paste0("running iteration#", iteration))
  predictions <- c()
  predictions <- predictClass(X, w, w0, K, N)

  y_predicted <- predictions[,1]
  predicted_prob <- predictions[,2:4]
  predicted_val <- predictions[,5]
  
  error <- 0
  for(i in 1:N){
    error <- error + log(predicted_prob[i, y_truth[i]] +  1e-100)
  }
  
  objective_values <- c(objective_values, -error)
  
  w_old <- w
  w0_old <- w0
  w <- w + eta * gradient_w(X, predicted_prob, y_truth)
  w0 <- w0 + eta * gradient_w0(predicted_prob, y_truth)
  if (sqrt((w0[1,2] - w0_old[1,2])^2 + (w0[1,1] - w0_old[1,1])^2 + (w0[1,3] - w0_old[1,3])^2) < epsilon) {
    break
  }
  
  iteration <- iteration + 1
}
colnames(w) <- c("w1", "w2", "w3")
print(w)
colnames(w0) <- c("w10", "w20", "w30")
rownames(w0) <- ""
print(w0)

# plot objective function during iterations
plot(1:(iteration), objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

# calculate confusion matrix
confusion_matrix <- table(y_predicted, y_truth)
print(confusion_matrix)

# evaluate discriminat function on a grid
x1_interval <- seq(from = -6, to = +6, by = 0.06)
x2_interval <- seq(from = -6, to = +6, by = 0.06)
x1_grid <- matrix(x1_interval, nrow = length(x1_interval), ncol = length(x1_interval), byrow = FALSE)
x2_grid <- matrix(x2_interval, nrow = length(x2_interval), ncol = length(x2_interval), byrow = TRUE)

discriminant_values <- matrix(0, c(length(x1_interval), length(x2_interval)))
f1 <- function(x1, x2) {
  if(max(softmax(cbind(x1, x2), w, w0, K, 1), softmax(cbind(x1, x2), w, w0, K, 2), softmax(cbind(x1, x2), w, w0, K, 3)) == softmax(cbind(x1, x2), w, w0, K, 1))
    return(1)
  if(max(softmax(cbind(x1, x2), w, w0, K, 1), softmax(cbind(x1, x2), w, w0, K, 2), softmax(cbind(x1, x2), w, w0, K, 3)) == softmax(cbind(x1, x2), w, w0, K, 2))
    return(2)
  else 
    return(3)  
  }
discriminant_values <- matrix(mapply(f1, x1_grid, x2_grid), nrow(x2_grid), ncol(x2_grid))

plot(X[y_truth == 1, 1], X[y_truth == 1, 2], type = "p", pch = 19, 
     col = rgb(0.2,0.5,0.2,0.9), xlim = c(-6, +6), ylim = c(-6, +6), xlab = "x1", ylab = "x2", las = 1)
points(X[y_truth == 2, 1], X[y_truth == 2, 2], type = "p", pch = 19, col = rgb(0.9,0.4,0.1,0.9))
points(X[y_truth == 3, 1], X[y_truth == 3, 2], type = "p", pch = 19, col = rgb(0.1,0.4,0.7,0.9))
points(X[y_predicted != y_truth, 1], X[y_predicted != y_truth, 2], cex = 1.5, lwd = 2)
points(x1_grid[discriminant_values == 1], x2_grid[discriminant_values == 1], col = rgb(0.2,0.5,0.5,0.03), pch = 16)
points(x1_grid[discriminant_values == 2], x2_grid[discriminant_values == 2], col = rgb(0.9,0.4,0.1,0.03), pch = 16)
points(x1_grid[discriminant_values == 3], x2_grid[discriminant_values == 3], col = rgb(0.1,0.4,0.7,0.03), pch = 16)
contour(x1_interval, x2_interval, discriminant_values, levels = c(3), add = TRUE, lwd = 2, drawlabels = FALSE)
contour(x1_interval, x2_interval, discriminant_values, levels = c(2), add = TRUE, lwd = 2, drawlabels = FALSE)
