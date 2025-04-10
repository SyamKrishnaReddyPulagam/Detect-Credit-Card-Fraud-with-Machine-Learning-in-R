
# Load required packages
library(tidyverse)
library(caret)
library(data.table)
library(ROSE)       # For balancing the dataset
library(randomForest)
library(pROC)       # For ROC curve

# Load the dataset
data <- fread("creditcard.csv")

# Check structure and class imbalance
str(data)
table(data$Class)   # 0 = Not Fraud, 1 = Fraud

# Convert 'Class' to factor
data$Class <- as.factor(data$Class)

# Check for missing values
sum(is.na(data))

# Normalize 'Amount' and drop 'Time'
data$Amount <- scale(data$Amount)
data$Time <- NULL

# Split into training and test sets (80-20 split)
set.seed(123)
trainIndex <- createDataPartition(data$Class, p = .8, 
                                  list = FALSE, 
                                  times = 1)
trainData <- data[trainIndex, ]
testData  <- data[-trainIndex, ]

# Handle class imbalance using ROSE (oversampling)
balanced_train <- ROSE(Class ~ ., data = trainData, seed = 1)$data
table(balanced_train$Class)

# Train Random Forest model
set.seed(123)
rf_model <- randomForest(Class ~ ., data = balanced_train, ntree = 100)
print(rf_model)

# Predictions
predictions <- predict(rf_model, newdata = testData)

# Confusion Matrix
conf_matrix <- confusionMatrix(predictions, testData$Class)
print(conf_matrix)

# ROC Curve & AUC
rf_probs <- predict(rf_model, newdata = testData, type = "prob")[,2]
roc_curve <- roc(testData$Class, rf_probs)
plot(roc_curve, col = "blue", main = "ROC Curve")
auc(roc_curve)
