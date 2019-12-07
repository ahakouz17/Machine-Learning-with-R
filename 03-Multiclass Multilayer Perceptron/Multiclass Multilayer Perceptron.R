##################################################################################
###       COMP/INDR 421/521 INTRODUCTION TO MACHINE LEARNING (Fall 2017)       ###
###                HW03: Multiclass Multilayer Perceptron                      ###      
###                Developed by: Asma Hakouz                                   ### 
##################################################################################
library(MASS)
## GENERATING DATA
# choose an arbitrary value for the seed which will be used for random numbers generation
set.seed(401)

# mean parameters
class_means <- matrix(c(+2.0, +2.0,
                        -4.0, -4.0,
                        -2.0, +2.0,
                        +4.0, -4.0,
                        -2.0, -2.0,
                        +4.0, +4.0,
                        +2.0, -2.0,
                        -4.0, +4.0), 2, 8)
# covariance parameters
class_covariances <- array(c(+0.8, -0.6, -0.6, +0.8,
                             +0.4, +0.0, +0.0, +0.4,
                             +0.8, +0.6, +0.6, +0.8,
                             +0.4, +0.0, +0.0, +0.4,
                             +0.8, -0.6, -0.6, +0.8,
                             +0.4, +0.0, +0.0, +0.4,
                             +0.8, +0.6, +0.6, +0.8,
                             +0.4, +0.0, +0.0, +0.4), c(2, 2, 8))

# sample sizes
class_sizes <- c(100, 100, 100, 100)

# generate random samples
points1 <- mvrnorm(n = class_sizes[1] / 2, mu = class_means[,1], Sigma = class_covariances[,,1])
points2 <- mvrnorm(n = class_sizes[1] / 2, mu = class_means[,2], Sigma = class_covariances[,,2])
points3 <- mvrnorm(n = class_sizes[2] / 2, mu = class_means[,3], Sigma = class_covariances[,,3])
points4 <- mvrnorm(n = class_sizes[2] / 2, mu = class_means[,4], Sigma = class_covariances[,,4])
points5 <- mvrnorm(n = class_sizes[3] / 2, mu = class_means[,5], Sigma = class_covariances[,,5])
points6 <- mvrnorm(n = class_sizes[3] / 2, mu = class_means[,6], Sigma = class_covariances[,,6])
points7 <- mvrnorm(n = class_sizes[4] / 2, mu = class_means[,7], Sigma = class_covariances[,,7])
points8 <- mvrnorm(n = class_sizes[4] / 2, mu = class_means[,8], Sigma = class_covariances[,,8])
X <- rbind(points1, points2, points3, points4, points5, points6, points7, points8)
colnames(X) <- c("x1", "x2")


# generate corresponding labels
y <- c(rep(1, class_sizes[1]), rep(2, class_sizes[2]), rep(3, class_sizes[3]), rep(4, class_sizes[4]))

# write data to a file
write.csv(x = cbind(X, y), file = "HW3_data_set.csv", row.names = FALSE)

# plot data points generated
plot(points1[,1], points1[,2], type = "p", pch = 19, col = "red", las = 1,
     xlim = c(-6, 6), ylim = c(-6, 6),
     xlab = "x1", ylab = "x2")
points(points2[,1], points2[,2], type = "p", pch = 19, col = "red")
points(points3[,1], points3[,2], type = "p", pch = 19, col = "blue")
points(points4[,1], points4[,2], type = "p", pch = 19, col = "blue")
points(points5[,1], points5[,2], type = "p", pch = 19, col = "green")
points(points6[,1], points6[,2], type = "p", pch = 19, col = "green")
points(points7[,1], points7[,2], type = "p", pch = 19, col = "magenta")
points(points8[,1], points8[,2], type = "p", pch = 19, col = "magenta")



##################################################################################
## DATA PROCESSING AND ALGORITHM
# read data into memory
data_set <- read.csv("HW3_data_set.csv")

# get x1, x2 and y values
x1 <- data_set$x1
x2 <- data_set$x2
X <- cbind(x1, x2)
y_truth <- data_set$y


# get number of samples and number of features and number of classes
N <- length(y_truth)
D <- ncol(X)
K <- max(y_truth)
####################

###################
# set learning parameters
eta <- 0.1
epsilon <- 1e-3
H <- 20 # num of hidden layes perceptrons (excluding the bias)
max_iteration <- 200 # shouldnt be big because of overfitting

# define the sigmoid function
sigmoid <- function(a) {
  return (1 / (1 + exp(-a)))
}

###################
# randomly initalize W and v
set.seed(421)
W <- matrix(runif((D + 1) * H, min = -0.01, max = 0.01), D + 1, H)
v <- matrix(runif((H + 1) * K, min = -0.01, max = 0.01), H + 1, K)
####################

# define the softmax function
softmax <- function(Z, v, k, c){
  denomSum <- 0
  for(i in 1:k){
    denomSum <- denomSum + exp(Z %*% v[,i])
    
  }

  return (exp(Z %*% v[,c])/denomSum)
}

Z <- sigmoid(cbind(1, X) %*% W)
y_predicted <- sapply(1:K, function(c){softmax(cbind(rep(1, N), Z), v, K, c)})
last_v <- c()
last_w <- c()

error <- 0
for(i in 1:N){
  error <- error + log(y_predicted[i, y_truth[i]] +  1e-100)
}
objective_values <- -error
########################
 
# learn W and v using gradient descent and online learning
iteration <- 1
while (1) {
  print(paste0("running iteration#", iteration))
  for (i in sample(N)) { # sample(N) gives a random ordering of numbers 1 to N, we want to access data in a random order
    # calculate hidden nodes
    Z[i,] <- sigmoid(c(1, X[i,]) %*% W)

    # calculate output node
    y_predicted[i,] <- sapply(1:K, function(c){softmax(matrix(c(1, Z[i,]), 1, H + 1), v, K, c)})
    for(e in 1:K){
      if(y_truth[i] == e){
        v[,e] <- v[,e] + eta * (1 - y_predicted[i, y_truth[i]]) * c(1, Z[i,])
      } else {
        v[,e] <- v[,e] + eta * -1 * y_predicted[i, e] * c(1, Z[i,])  
      }
    }
    for (h in 1:H) {
      sum_error <- 0
      for(class in 1:K){
        if(y_truth[i] == class){
          sum_error <- sum_error + (1 - y_predicted[i, class]) * v[h, class]
        } else {
          sum_error <- sum_error - y_predicted[i, class] * v[h, class]
        }
      }
      W[,h] <- W[,h] + eta * sum_error * Z[i, h] * (1 - Z[i, h]) * c(1, X[i,])
    }
  }
  
  error <- 0
  for(i in 1:N){
    error <- error + log(y_predicted[i, y_truth[i]] +  1e-100)
  }
  objective_values <- c(objective_values, -error)
  if(iteration != 1){
    if ((abs(objective_values[iteration] - objective_values[iteration - 1]) < epsilon || iteration >= max_iteration)) {
      break
    }
  }
  last_v <- v
  last_w <- W
  iteration <- iteration + 1
}
print(W)
print(v)

# plot objective function during iterations
plot(1:(iteration + 1), objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

# calculate confusion matrix
y_predicted_class <- c()
for(i in 1:N){
  y_predicted_class <- rbind(y_predicted_class, which.max(y_predicted[i,]))
}
confusion_matrix <- table(y_predicted_class, y_truth)
print(confusion_matrix)

# evaluate discriminat function on a grid
x1_interval <- seq(from = -6, to = +6, by = 0.06)
x2_interval <- seq(from = -6, to = +6, by = 0.06)
x1_grid <- matrix(x1_interval, nrow = length(x1_interval), ncol = length(x1_interval), byrow = FALSE)
x2_grid <- matrix(x2_interval, nrow = length(x2_interval), ncol = length(x2_interval), byrow = TRUE)

f <- function(x1, x2, K, N, Z) {
  Z <- sigmoid(cbind(1, x1, x2) %*% last_w)
  y_predicted_i <- sapply(1:K, function(c){softmax(cbind(rep(1, N), Z), last_v, K, c)})
  
  if(which.max(y_predicted_i) == 1)
    return(1)
  if(which.max(y_predicted_i) == 2)
    return(2)
  if(which.max(y_predicted_i) == 3)
    return(3)
  else 
    return(4)  
}
discriminant_values <- matrix(mapply(f, x1_grid, x2_grid, K, N, Z), nrow(x2_grid), ncol(x2_grid))

plot(X[y_truth == 1, 1], X[y_truth == 1, 2], type = "p", pch = 19, col = "red",
     xlim = c(-6, +6),
     ylim = c(-6, +6),
     xlab = "x1", ylab = "x2", las = 1)
points(X[y_truth == 2, 1], X[y_truth == 2, 2], type = "p", pch = 19, col = "blue")
points(X[y_truth == 3, 1], X[y_truth == 3, 2], type = "p", pch = 19, col = "green")
points(X[y_truth == 4, 1], X[y_truth == 4, 2], type = "p", pch = 19, col = "magenta")

points(X[y_predicted_class != y_truth, 1], X[y_predicted_class != y_truth, 2], cex = 1.5, lwd = 2)
points(x1_grid[discriminant_values == 1], x2_grid[discriminant_values == 1], col = rgb(red = 1, green = 0, blue = 0, alpha = 0.01), pch = 16)
points(x1_grid[discriminant_values == 2], x2_grid[discriminant_values == 2], col = rgb(red = 0, green = 0, blue = 1, alpha = 0.01), pch = 16)
points(x1_grid[discriminant_values == 3], x2_grid[discriminant_values == 3], col = rgb(red = 0, green = 1, blue = 0, alpha = 0.01), pch = 16)
points(x1_grid[discriminant_values == 4], x2_grid[discriminant_values == 4], col = rgb(red = 0.5, green = 0, blue = 0.5, alpha = 0.01), pch = 16)

contour(x1_interval, x2_interval, discriminant_values, levels = c(3), add = TRUE, lwd = 2, drawlabels = FALSE)
contour(x1_interval, x2_interval, discriminant_values, levels = c(2), add = TRUE, lwd = 2, drawlabels = FALSE)
contour(x1_interval, x2_interval, discriminant_values, levels = c(4), add = TRUE, lwd = 2, drawlabels = FALSE)
contour(x1_interval, x2_interval, discriminant_values, levels = c(1), add = TRUE, lwd = 2, drawlabels = FALSE)

