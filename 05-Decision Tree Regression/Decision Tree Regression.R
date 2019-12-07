########### Data pre-processing and visualization########
# read data into memory
data_set <- read.csv("hw05_data_set.csv")

# get x and y values
x <- data_set$x
y <- data_set$y

set.seed(521)
# get number of samples
N <- length(y)
train_sample_count = 100

# randomly dividing the data set to training and test samples
train_indices <-  c(sample(1:N, floor(train_sample_count)))
X_train <- x[train_indices]
y_train <- y[train_indices]
X_test <- x[-train_indices] 
y_test <- y[-train_indices]

plot(X_train, y_train, type = "p", pch = 19, col = "blue", 
     main = "Data Visualization", xlab = "X", ylab = "Y")
points(X_test, y_test, type = "p", pch = 19, col = "red")
legend("topright", legend = c("training", "test"), col = c("blue", "red"), pch = c(19, 19), 
       bty = "n", pt.cex = 0.7, cex = 0.7, text.col = rgb(0.3,0.3,0.3,1), horiz = F, inset = c(0.01, 0.01))

# get numbers of train and test samples
N_train <- length(y_train)
N_test <- length(y_test)
############ Initialization ######################
# create necessary data structures
node_indices <- list()
is_terminal <- c()
need_split <- c()
node_splits <- c()      # the threshold    (Xj > Wmo)
node_frequencies <- list()

# put all training instances into the root node
node_indices <- list(1:N_train)
is_terminal <- c(FALSE)
need_split <- c(TRUE)

### UNCOMMENT THIS TO GET THE OUTPUT FROM THE USER ######
#p <- as.integer(readline(prompt="Enter the pre-pruning parameter: "))

p <- 10
############ Learning ################
# learning algorithm
while (1) {
  # find nodes that need splitting
  split_nodes <- which(need_split)
  # check whether we reach all terminal nodes
  if (length(split_nodes) == 0) {
    break
  }
  # find best split positions for all nodes
  for (split_node in split_nodes) {
    data_indices <- node_indices[[split_node]]
    need_split[split_node] <- FALSE
    # check whether node is pure
    if (length(data_indices) <= p) {  # if all data point are all from 1 class
      is_terminal[split_node] <- TRUE  # if all point are from the same class, then no need to split
      node_frequencies[[split_node]] <- mean(y_train[data_indices], na.rm = TRUE)
    } else {
      is_terminal[split_node] <- FALSE # 
      
      unique_values <- sort(unique(X_train[data_indices], na.rm = TRUE))  # to decide what are the possible splits
      split_positions <- (unique_values[-1] + unique_values[-length(unique_values)]) / 2   
      split_scores <- rep(1000000, length(split_positions))
      for (s in 1:length(split_positions)) {
        left_indices <- data_indices[which(X_train[data_indices] <= split_positions[s])]
        right_indices <- data_indices[which(X_train[data_indices] > split_positions[s])]
        left_avg <- mean(y_train[left_indices], na.rm = TRUE)
        right_avg <- mean(y_train[right_indices], na.rm = TRUE)
        if(is.na(left_avg)){
          left_avg <- 0
        }
        if(is.na(right_avg)){
          right_avg <- 0
        }
        split_scores[s] <- 1 / length(data_indices) * (sum((y_train[left_indices] - left_avg)**2, na.rm = TRUE)
                                                       + sum((y_train[right_indices] - right_avg)**2, na.rm = TRUE))
        if(1 / length(data_indices) * (sum((y_train[left_indices] - left_avg)**2, na.rm = TRUE)
                                       + sum((y_train[right_indices] - right_avg)**2, na.rm = TRUE)) == 0){
          split_scores[s] <- 100000
        }
      }
     
      best_scores <- min(split_scores)
      best_splits <- split_positions[which.min(split_scores)]

      # decide where to split on which feature
      node_splits[split_node] <- best_splits
      
      # create left node using the selected split
      if(length(left_indices) != 0){
        left_indices <- data_indices[which(X_train[data_indices] <= best_splits)]
        node_indices[[2 * split_node]] <- left_indices
        is_terminal[2 * split_node] <- FALSE
        need_split[2 * split_node] <- TRUE
      }
      
      # create left node using the selected split
      if(length(right_indices) != 0){
        right_indices <- data_indices[which(X_train[data_indices] > best_splits)]
        node_indices[[2 * split_node + 1]] <- right_indices
        is_terminal[2 * split_node + 1] <- FALSE
        need_split[2 * split_node + 1] <- TRUE
      }
    }
  }
}

###### extract rules ##################
terminal_nodes <- which(is_terminal)
for (terminal_node in terminal_nodes) {
  index <- terminal_node
  rules <- c()
  while (index > 1) {
    parent <- floor(index / 2)
    if (index %% 2 == 0) {
      # if node is left child of its parent
      rules <- c(sprintf("x < %g", node_splits[parent]), rules)
    } else {
      # if node is right child of its parent
      rules <- c(sprintf("x >= %g", node_splits[parent]), rules)
    }
    index <- parent
  }
  print(sprintf("{%s} => [%s]", paste0(rules, collapse = " AND "), paste0(node_frequencies[[terminal_node]], collapse = "-")))
}

x_interval <- seq(from = 0, to = 60, by = 0.01)
y_int <- rep(0, length(x_interval))
for (i in 1:length(x_interval)) {
  index <- 1
  while (1) {
    if (is_terminal[index] == TRUE) {
      y_int[i] <- node_frequencies[[index]]
      break
    } else if(is_terminal[index] == FALSE) {
      if (x_interval[i] <= node_splits[index]) {
        index <- index * 2
      } else {
        index <- index * 2 + 1
      }
    }
  }
}
lines(x_interval, y_int, lwd = 2, col = "black") 
#################################

### traverse tree for test data points ####
y_predicted <- rep(0, N_test)
for (i in 1:N_test) {
  index <- 1
  while (1) {
    if (is_terminal[index] == TRUE) {
      y_predicted[i] <- node_frequencies[[index]]
      break
    } else {
      
      print("----------")
      print(index)
      print(node_splits[index])
      if (X_test[i] < node_splits[index]) {
        index <- index * 2
      } else {
        index <- index * 2 + 1
      }
    }
  }
}

rmse = sqrt(mean((y_test - y_predicted)**2))
sprintf("RMSE is %f when P is %d", rmse, p)
######################################

####################################
rmse_values = c()
p_range <- 3:20

for(prun in p_range){
  node_indices <- list()
  is_terminal <- c()
  need_split <- c()
  node_splits <- c()      # the threshold    (Xj > Wmo)
  node_frequencies <- list()
  
  # put all training instances into the root node
  node_indices <- list(1:N_train)
  is_terminal <- c(FALSE)
  need_split <- c(TRUE)
  # learning algorithm
  while (1) {
    # find nodes that need splitting
    split_nodes <- which(need_split)
    # check whether we reach all terminal nodes
    if (length(split_nodes) == 0) {
      break
    }
    # find best split positions for all nodes
    for (split_node in split_nodes) {
      data_indices <- node_indices[[split_node]]
      need_split[split_node] <- FALSE
      # check whether node is pure
      if (length(data_indices) <= prun) {  # if all data point are all from 1 class
        is_terminal[split_node] <- TRUE  # if all point are from the same class, then no need to split
        node_frequencies[[split_node]] <- mean(y_train[data_indices], na.rm = TRUE)
      } else {
        is_terminal[split_node] <- FALSE # 
        
        unique_values <- sort(unique(X_train[data_indices], na.rm = TRUE))  # to decide what are the possible splits
        split_positions <- (unique_values[-1] + unique_values[-length(unique_values)]) / 2   
        split_scores <- rep(0, length(split_positions))
        for (s in 1:length(split_positions)) {
          left_indices <- data_indices[which(X_train[data_indices] <= split_positions[s])]
          right_indices <- data_indices[which(X_train[data_indices] > split_positions[s])]
          left_avg <- mean(y_train[left_indices], na.rm = TRUE)
          right_avg <- mean(y_train[right_indices], na.rm = TRUE)
          
          split_scores[s] <- 1 / length(data_indices) * (sum((y_train[left_indices] - left_avg)**2, na.rm = TRUE)
                                                         + sum((y_train[right_indices] - right_avg)**2, na.rm = TRUE))
        }
        best_scores <- min(split_scores)
        best_splits <- split_positions[which.min(split_scores)]
        
        # decide where to split on which feature
        node_splits[split_node] <- best_splits
        
        # create left node using the selected split
        if(length(left_indices) != 0){
          left_indices <- data_indices[which(X_train[data_indices] < best_splits)]
          node_indices[[2 * split_node]] <- left_indices
          is_terminal[2 * split_node] <- FALSE
          need_split[2 * split_node] <- TRUE
        }
        
        # create left node using the selected split
        if(length(left_indices) != 0){
          right_indices <- data_indices[which(X_train[data_indices] >= best_splits)]
          node_indices[[2 * split_node + 1]] <- right_indices
          is_terminal[2 * split_node + 1] <- FALSE
          need_split[2 * split_node + 1] <- TRUE
        }
      }
    }
  }
  # traverse tree for test data points
  y_predicted <- rep(0, N_test)
  for (i in 1:N_test) {
    index <- 1
    while (1) {
      if (is_terminal[index] == TRUE) {
        y_predicted[i] <- node_frequencies[[index]]
        break
      } else {
        if (X_test[i] <= node_splits[index]) {
          index <- index * 2
        } else {
          index <- index * 2 + 1
        }
      }
    }
  }
  
  rmse = sqrt(mean((y_test - y_predicted)**2))
  sprintf("RMSE is %f when P is %d", rmse, prun)
  rmse_values <- c(rmse_values, rmse)
}
plot(p_range, rmse_values)
lines(p_range, rmse_values)

