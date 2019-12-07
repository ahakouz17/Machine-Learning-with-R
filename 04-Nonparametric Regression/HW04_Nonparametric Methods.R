# read data into memory
data_set <- read.csv("hw04_data_set.csv")

# get x and y values
x <- data_set$x
y <- data_set$y

# get number of samples
N <- length(y)

train_sample_count = 100

# randomly dividing the data set to training and test samples
train_indices <-  c(sample(1:N, floor(train_sample_count)))
X_train <- x[train_indices]
y_train <- y[train_indices]
X_test <- x[-train_indices]  # - means excluding
y_test <- y[-train_indices]

plot(X_train, y_train, type = "p", pch = 19, col = "blue", 
       main = "Data Visualization", xlab = "X", ylab = "Y")
points(X_test, y_test, type = "p", pch = 19, col = "red")

minimum_value <- min(x) - 2.4
maximum_value <- max(x) + 2.4
data_step = 0.01
data_interval <- seq(from = minimum_value, to = maximum_value, by = data_step)

legend("topright", 
       legend = c("training", "test"), 
       col = c("blue", 
               "red"), 
       pch = c(19, 19), bty = "n", pt.cex = 1, cex = 1, 
       text.col = rgb(0.3,0.3,0.3,1), horiz = F, inset = c(0.01, 0.01))

############## 1 - regressogram estimator #########################
### Initialization
bin_width <- 3

left_borders <- seq(from = minimum_value, to = maximum_value - bin_width, by = bin_width)
right_borders <- seq(from = minimum_value + bin_width, to = maximum_value, by = bin_width)

### Calculation
p_head <- sapply(1:length(left_borders), function(b) 
    {sum(y_train[(left_borders[b] < X_train & X_train <= right_borders[b])]) /
      (sum(left_borders[b] < X_train & X_train <= right_borders[b]) + 1e-03)})

### Visualization
plot(X_train, y_train, type = "p", pch = 19, col = "blue", ylab = "density", xlab = "x")
points(X_test, y_test, type = "p", pch = 19, col = "red")

for (b in 1:length(left_borders)) {
  lines(c(left_borders[b], right_borders[b]), c(p_head[b], p_head[b]), lwd = 2, col = "black")
  if (b < length(left_borders)) {
    lines(c(right_borders[b], right_borders[b]), c(p_head[b], p_head[b + 1]), lwd = 2, col = "black") 
  }
}

legend("topright", 
       legend = c("training", "test"), 
       col = c("blue", 
               "red"), 
       pch = c(19, 19), bty = "n", pt.cex = 1, cex = 1, 
       text.col = rgb(0.3,0.3,0.3,1), horiz = F, inset = c(0.01, 0.01))

### Evaluation - RMSE calculation
rmse = sqrt(mean((y_test - p_head[ceiling(abs(X_test - minimum_value) / bin_width)]) ** 2))

sprintf("Regressogram => RMSE is %f when h is 3", rmse)

############## 2 - running mean smoother #########################
### Initialization
bin_width <- 3
p_head <- sapply(data_interval, function(x) 
  {sum(y_train[(x - 0.5 * bin_width) < X_train & X_train <= (x + 0.5 * bin_width)]) / 
    (sum((x - 0.5 * bin_width) < X_train & X_train <= (x + 0.5 * bin_width))+ 1e-03)})

### Visualization
plot(X_train, y_train, type = "p", pch = 19, col = "blue", ylab = "density", xlab = "x")
points(X_test, y_test, type = "p", pch = 19, col = "red")
lines(data_interval, p_head, type = "l", lwd = 2, col = "black")

legend("topright", 
       legend = c("training", "test"), 
       col = c("blue", 
               "red"), 
       pch = c(19, 19), bty = "n", pt.cex = 1, cex = 1, 
       text.col = rgb(0.3,0.3,0.3,1), horiz = F, inset = c(0.01, 0.01))

### Evaluation - RMSE calculation
rmse = sqrt(mean((y_test - p_head[ceiling(X_test/data_step)]) ** 2))
sprintf("Running mean smoother => RMSE is %f when h is 3", rmse)

############## 3 - Kernel smoother########################################
bin_width <- 1
p_head <- sapply(data_interval, function(x) 
  {sum((1 / sqrt(2 * pi)) * exp(-0.5 * ((x - X_train)/ bin_width) ** 2) * y_train) / 
    (sum((1 / sqrt(2 * pi)) * exp(-0.5 * ((x - X_train)/ bin_width) ** 2)))})

plot(X_train, y_train, type = "p", pch = 19, col = "blue", ylab = "density", xlab = "x")
points(X_test, y_test, type = "p", pch = 19, col = "red")
lines(data_interval, p_head, type = "l", lwd = 2, col = "black")

legend("topright", legend = c("training", "test"), 
       col = c("blue", "red"), pch = c(19, 19), bty = "n", pt.cex = 1, cex = 1, 
       text.col = rgb(0.3,0.3,0.3,1), horiz = F, inset = c(0.01, 0.01))

rmse = sqrt(mean((y_test - p_head[ceiling(X_test/0.01)]) ** 2))

## RMSE calculation
sprintf("Kernel Smoother => RMSE is %f when h is 1", rmse)

trials <- 20
### avg RMSE for regressogram
sum_rmse <- 0
bin_width <- 3
for(f in 1:(trials*100)){
  # randomly dividing the data set to training and test samples
  train_indices <-  c(sample(1:N, floor(train_sample_count)))
  X_train <- x[train_indices]
  y_train <- y[train_indices]
  X_test <- x[-train_indices]  # - means excluding
  y_test <- y[-train_indices]
  
  ### Calculation
  p_head <- sapply(1:length(left_borders), function(b) 
  {sum(y_train[(left_borders[b] < X_train & X_train <= right_borders[b])]) /
      (sum(left_borders[b] < X_train & X_train <= right_borders[b]) + 1e-03)})
  ### Evaluation - RMSE calculation
  rmse = sqrt(mean((y_test - p_head[ceiling(abs(X_test - minimum_value) / bin_width)]) ** 2))
  sum_rmse <- sum_rmse + rmse
}
average_rmse <- sum_rmse/(trials*100);
sprintf("Regressogram => Average RMSE is %f when h is 3", print(average_rmse))


### avg RMSE for running mean smoother
sum_rmse <- 0
bin_width <- 3
for(f in 1:trials){
  # randomly dividing the data set to training and test samples
  train_indices <-  c(sample(1:N, floor(train_sample_count)))
  X_train <- x[train_indices]
  y_train <- y[train_indices]
  X_test <- x[-train_indices]  # - means excluding
  y_test <- y[-train_indices]
  
  ### Calculation
  p_head <- sapply(data_interval, function(x) {sum(y_train[(x - 0.5 * bin_width) < X_train & X_train <= (x + 0.5 * bin_width)]) / (sum((x - 0.5 * bin_width) < X_train & X_train <= (x + 0.5 * bin_width))+ 1e-03)})
  
  ### Evaluation - RMSE calculation
  rmse = sqrt(mean((y_test - p_head[ceiling(X_test/data_step)]) ** 2))
  sum_rmse <- sum_rmse + rmse
}
average_rmse <- sum_rmse/trials;
sprintf("Running mean smoother => Average RMSE is %f when h is 3", print(average_rmse))

### avg RMSE for kernel smoother
sum_rmse <- 0
bin_width <- 1
for(f in 1:trials){
  # randomly dividing the data set to training and test samples
  train_indices <-  c(sample(1:N, floor(train_sample_count)))
  X_train <- x[train_indices]
  y_train <- y[train_indices]
  X_test <- x[-train_indices]  # - means excluding
  y_test <- y[-train_indices]
  
  ### Calculation
  p_head <- sapply(data_interval, function(x) 
    {sum((1 / sqrt(2 * pi)) * exp(-0.5 * ((x - X_train)/ bin_width) ** 2) * y_train) / (sum((1 / sqrt(2 * pi)) * exp(-0.5 * ((x - X_train)/ bin_width) ** 2)))})
  
  ### Evaluation - RMSE calculation
  rmse = sqrt(mean((y_test - p_head[ceiling(X_test/0.01)]) ** 2))
  sum_rmse <- sum_rmse + rmse
}
average_rmse <- sum_rmse/trials;
sprintf("Kernel smoother => Average RMSE is %f when h is 1", print(average_rmse))
