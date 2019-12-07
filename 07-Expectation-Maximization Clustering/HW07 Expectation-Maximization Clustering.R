##################################################################################
###       COMP/INDR 421/521 INTRODUCTION TO MACHINE LEARNING (Fall 2017)       ###
###                HW07: Expectation-Maximization Clustering                   ###      
###                Developed by: Asma Hakouz                                   ### 
##################################################################################
library(MASS)
## GENERATING DATA
# choose an arbitrary value for the seed which will be used for random numbers generation
set.seed(521)
# mean parameters
cluster_means <- array(c(2.5, 2.5, 
                   -2.5, 2.5,
                   -2.5, -2.5,
                   2.5, -2.5,
                   0, 0), c(1, 2, 5))
# Coveriance matrices
cluster_sigma <- array(c(+0.8, -0.6, -0.6, +0.8,
                         +0.8, +0.6, +0.6, +0.8,
                         +0.8, -0.6, -0.6, +0.8,
                         +0.8, +0.6, +0.6, +0.8,
                         +1.6, 0.0, 0.0, +1.6), c(2, 2, 5))
# sample sizes
cluster_sizes <- c(50, 50, 50, 50, 100)

# generate random samples from multivariate (in our case it's bivariate) normal distributions
points <- c()
for(i in 1:5){
  points <- rbind(points, MASS::mvrnorm(n = cluster_sizes[i], cluster_means[,, i], cluster_sigma[,, i]))
}
  
# plot the generated data points.  
plot(points[,1], points[,2], type = "p", col = rgb(0.2,0.4,0.1,0.9), lwd = 0.5, 
     xlab = "x1", ylab = "x2", ylim = c(-6, max(points[,2])),
     xlim = c(-6, max(points[,1])), pch = 19)
X <- points
N <- 300
K <- 5

# K-means initialization
centroids <<- NULL
assignments <<- NULL
for(i in 1:2){
  if (is.null(centroids) == TRUE) {
    # for the first iteration; random initialization of centroids
    centroids <- X[sample(1:N, K),]
  } else {
    for (k in 1:K) {
      centroids[k,] <- colMeans(X[assignments == k,])
    }  
  }
  #update_memberships,
  # check dist between all DPs and centroids
  D <- as.matrix(dist(rbind(centroids, X), method = "euclidean"))
  D <- D[1:nrow(centroids), (nrow(centroids) + 1):(nrow(centroids) + nrow(X))]
  #assignments <<- apply(D, MARGIN = 2, FUN = function(x) sort(x, index.return = TRUE)$ix[1])
  assignments <<- sapply(1:N, function(i) {which.min(D[,i])})
  
  colors <- c("#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928", "#a6cee3",
              "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99")
  if (is.null(X) == FALSE && is.null(assignments) == TRUE) {
    par(mar = c(0, 0, 0, 0), oma = c(0, 0, 0, 0))
    plot(X[,1], X[,2], col = "lightgray", xlim = c(-6, 6), ylim = c(-6, 6), xlab = "", ylab = "",
         cex = 1.5, axes = FALSE, pch = 19)
  }
  if (is.null(X) == FALSE && is.null(assignments) == FALSE) {
    par(mar = c(0, 0, 0, 0), oma = c(0, 0, 0, 0))
    plot(X[,1], X[,2], col = colors[assignments], xlim = c(-6, 6), ylim = c(-6, 6), xlab = "",
         ylab = "", cex = 1.5, axes = FALSE, pch = 19)
  }
  if (is.null(centroids) == FALSE) {
    points(centroids[,1], centroids[,2], col = "black", pch = 19, cex = 3) 
  }
}

# estimate classes variances
sample_covariance <- array(sapply(X = 1:K, FUN = function(c) {
  matrix(c(mean((X[assignments == c, 1] - centroids[c, 1])^2), 
           mean((X[assignments == c, 1] - centroids[c, 1])*(X[assignments == c, 2] - centroids[c, 2])),
           mean((X[assignments == c, 1] - centroids[c, 1])*(X[assignments == c, 2] - centroids[c, 2])),
           mean((X[assignments == c, 2] - centroids[c, 2])^2)))}), dim=c(2,2,K))

# calculate prior probabilities
class_priors <- sapply(X = 1:K, FUN = function(c) {mean(assignments == c)})

for(loop in 1:100){
  h_cluster <- sapply(1:N, function(t) {sapply(1:K, function(i) {
    (class_priors[i] * det(sample_covariance[,, i]) ** (-1/2.0) *
       exp(-1/2.0 *(t(as.matrix(X[t,] - centroids[i,])) %*% chol2inv(chol(sample_covariance[,, i])) %*%
                      as.matrix(X[t,] - centroids[i,]))))/
      sum(sapply(1:K, function(i) {
        (class_priors[i] * det(sample_covariance[,, i]) ** (-1/2.0) * 
           exp(-1/2.0 *(t(as.matrix(X[t,] - centroids[i,])) %*% chol2inv(chol(sample_covariance[,, i])) %*% 
                          as.matrix(X[t,] - centroids[i,])))) 
      }))
  })})
  
  class_priors <- sapply(1:K, function(j) {mean(h_cluster[j, ])})
  centroids <- t(sapply(1:k, function(j) {colSums(cbind(as.matrix(h_cluster[j,]), as.matrix(h_cluster[j,])) *
                                                    X)/ sum(h_cluster[j,])}))
  sample_covariance <- array(sapply(1:K, function(i){
    array(rowSums(sapply(1:N, function(t){
      h_cluster[i,t] * (X[t,] - centroids[i,]) %*% t(X[t,] - centroids[i,])
    })), c(2, 2)) / sum(h_cluster[i,])
  }), c(2, 2, K))
    
}  

par(mar = c(0, 0, 0, 0), oma = c(0, 0, 0, 0))
plot(X[,1], X[,2], col = colors[sapply(1:N, function(i){which.max(h_cluster[,i])})], xlim = c(-6, 6),
     ylim = c(-6, 6), xlab = "", ylab = "", cex = 1.5, axes = FALSE, pch = 19)

points(centroids[,1], centroids[,2], col = "black", pch = 19, cex = 3) 

print(centroids)

library(mvtnorm)
x.points <- seq(-6,6,length.out=100)
y.points <- x.points
for(j in 1:K){
  z <- matrix(0, nrow=100, ncol=100)
  mu <- centroids[j,]
  sigma <- sample_covariance[, , j]
  for (i in 1:100) {
    for (r in 1:100) {
      z[i,r] <- dmvnorm(c(x.points[i],y.points[r]),
                        mean=mu,sigma=sigma)
    }
  }
  contour(x.points,y.points, z, nlevels = 1, levels = c(0.05),add = TRUE, lwd = 2, drawlabels = FALSE,  lty = "solid")
  
}
for(j in 1:K){
  z <- matrix(0, nrow=100, ncol=100)
  mu <- cluster_means[,,j]
  sigma <- cluster_sigma[, , j]
  for (i in 1:100) {
    for (r in 1:100) {
      z[i,r] <- dmvnorm(c(x.points[i],y.points[r]),
                        mean=mu,sigma=sigma)
    }
  }
  contour(x.points,y.points, z, nlevels = 1, levels = c(0.05),add = TRUE, lwd = 2, drawlabels = FALSE,  lty = "dashed")
  
}
