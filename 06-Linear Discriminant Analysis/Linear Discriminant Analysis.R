# read testing and training data into memory
training_digits <- read.csv("hw06_mnist_training_digits.csv", header = FALSE)
training_labels <- read.csv("hw06_mnist_training_labels.csv", header = FALSE)

test_digits <- read.csv("hw06_mnist_test_digits.csv", header = FALSE)
test_labels <- read.csv("hw06_mnist_test_labels.csv", header = FALSE)

# get training set's X and y values
X <- as.matrix(training_digits) / 255
y <- training_labels[,1]

# get test set's X and y values
X_test <- as.matrix(test_digits) / 255
y_test <- test_labels[,1]

# get number of samples and number of features and number of classes
N <- length(y)
D <- ncol(X)
k <- 10
X_t <- t(X)

# calculate sample mean for each class
class_means <- matrix(0, D, k)
for(c in 1:k){
  class_means[,c] <- (sapply(X = 1:D, FUN = function(d) {mean(X_t[d, y == c])}))
}

# calculate the overall sample mean
overall_mean <- sapply(X = 1:D, FUN = function(d) {mean(class_means[d,])})

# calculate within-classes scatter matrix
within_scatter_sum <- matrix(0, D, D)
for(c in 1:k) {
  for(i in 1:N){
    if(y[i] == c){
      within_scatter_sum <- within_scatter_sum + 
          (X_t[,i] - class_means[c]) %*% t(X_t[,i] - class_means[c])
    }
  }
}

# perturbing the diagonal to ensure invertability
diag(within_scatter_sum) <- diag(within_scatter_sum) + 1e-10

# calculate number of samples of each class
N_class <- sapply(X = 1:k, FUN = function(c) {sum(y==c)})

# calculate the between-class scatter matrix
between_scatter_sum <- matrix(0, D, D)
for(c in 1:k) {
  between_scatter_sum <- between_scatter_sum + 
    N_class[c] * (class_means[c] - overall_mean) %*% t(class_means[c] - overall_mean)
}

# calculate the Sw^-1 * Sb matrix
Scatter_X <- chol2inv(chol(within_scatter_sum)) %*% between_scatter_sum

# calculate the eigenvalues and eigenvectors
decomposition <- eigen(Scatter_X, symmetric = TRUE)

# calculate two-dimensional projections
Z <- t(decomposition$vectors[,1:2]) %*% t(X)

# plot two-dimensional projections for training data
point_colors <- c("#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6")
plot(Z[1,], Z[2,], type = "p", pch = 19, col = point_colors[y], cex = 0,
     xlab = "Dimension 1", ylab = "Dimension 2", las = 1)
text(Z[1,], Z[2,], labels = y %% 10, col = point_colors[y])

Z_test <- t(decomposition$vectors[,1:2]) %*% t(X_test)
# plot two-dimensional projections for test data
point_colors <- c("#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6")
plot(Z_test[1,], Z_test[2,], type = "p", pch = 19, col = point_colors[y_test], cex = 0,
     xlab = "Dimension 1", ylab = "Dimension 2", las = 1)
text(Z_test[1,], Z_test[2,], labels = y_test %% 10, col = point_colors[y_test])

accuracy <- matrix(0, 9, 1)
  
# train the KNN classifier
for (r in 1:min(D, 9)) {
  Z_r <- (X_test) %*% decomposition$vectors[,1:r]
  # knn classification
  k_n <- 5 # number of neighbors for KNN
  k_c <- 10 # number of classes
  minimum_value <- min(Z_r)
  maximum_value <- max(Z_r)
  data_interval <- seq(from = minimum_value, to = maximum_value, by = 0.01)
  p_head <- matrix(0, length(Z_r), k_c)

  for (c in 1:10) {
    p_head[,c] <- sapply(1:N, function(x) 
      {sum(y_test[order(sapply(1:N, function(i)
        {dist(rbind(Z_r[x,], Z_r[i,]))}), decreasing = FALSE)[1:k]] == c) / k}) 
  }
  y_predicted <- sapply(1:N, function(i){(which.max(p_head[i, ])) %% 10})
  confusion_matrix <- table(y_predicted, y_test %% 10)
  sprintf("training and calculating accuracy for R = %d", print(r))
  print(confusion_matrix)
  accuracy[r] <- sum(diag(confusion_matrix))/N * 100
}

r_range = 1:9
plot(r_range, accuracy)
lines(r_range, accuracy)

# plot(1:min(D, 100), reconstruction_error[1:min(D, 100)], 
#      type = "l", las = 1, lwd = 2,
#      xlab = "R", ylab = "Average reconstruction error")
# abline(h = 0, lwd = 2, lty = 2, col = "blue")

# plot first 100 eigenvectors
# layout(matrix(1:100, 10, 10, byrow = TRUE))
# par(mar = c(0, 0, 0, 0), oma = c(0, 0, 0, 0))
# for (component in 1:100) {
#   image(matrix(decomposition$vectors[,component], nrow = 28)[,28:1], col = gray(12:1/12), axes = FALSE)
#   dev.next()
# }

# # plot scree graph
# plot(1:D, decomposition$values, 
#      type = "l", las = 1, lwd = 2,
#      xlab = "Eigenvalue index", ylab = "Eigenvalue")

# plot proportion of variance explained
# pove <- cumsum(decomposition$values) / sum(decomposition$values)
# plot(1:D, pove, 
#      type = "l", las = 1, lwd = 2,
#      xlab = "R", ylab = "Proportion of variance explained")
# abline(h = 0.95, lwd = 2, lty = 2, col = "blue")
# abline(v = which(pove > 0.95)[1], lwd = 2, lty = 2, col = "blue")

