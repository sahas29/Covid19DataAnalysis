#data sources: 
#Covid cases: https://github.com/RamiKrispin/coronavirus
#State data: r tidycensus package

install.packages("neuralnet")
library(neuralnet)
library(tidyverse)
library(forecast) # for accuracy() function
library(caret) # for preProcess() function
library(stats) # for kmeans

filter()
yearOne <- read.csv("Census2020.csv")
yearTwo <- read.csv("CensusDat2021.csv")
covid <- read.csv("us-states.csv")
covid

yearOneCovid <- filter(covid, date == as.Date("2020-12-31"))
yearOneCovid <- subset(yearOneCovid, select = -c(date))
yearOneCovid

yearTwoCovid <- filter(covid, date == as.Date("2021-12-31"))
yearTwoCovid <- subset(yearTwoCovid, select = -c(date))
yearTwoCovid

test <- data.frame(Link=yearTwoCovid$state,yearTwoCovid[3:4]-yearOneCovid[3:4])



yearTwoCovid$cases<-(yearTwoCovid$cases-yearOneCovid$cases)
yearTwoCovid

df <- read.csv("CensusFinal.csv")
df
# training/validation split
set.seed(12345)
train.proportion <- 0.7
validation.proportion <- 0.3

# pull out the training data
train.index <- sample(1:nrow(df), nrow(df)*train.proportion)
train.data <- df[train.index,]
validation.data <- df[-train.index,]

head(train.data)

normalizer <- preProcess(train.data, method="range")
train.norm <- predict(normalizer, train.data)
validation.norm <- predict(normalizer, validation.data)

nn.2 <- neuralnet(deaths_cases_ratio ~ median_age	+ precent_65_and_over +	poverty_rate + percent_uninsured	+ percent_bachelors_or_higher + median_household_income
, train.norm, hidden = 2, linear.output = TRUE, data=df)


nn.2$weights
plot(nn.2)

summary(nn.2)

validation.preds.norm.nn.2 <- predict(nn.2, newdata=validation.norm)
accuracy(validation.norm$deaths_cases_ratio, validation.preds.norm.nn.2)

nn.3 <- neuralnet(deaths_cases_ratio ~ median_age	+ precent_65_and_over +	poverty_rate + percent_uninsured	+ percent_bachelors_or_higher + median_household_income
                  , train.norm, hidden = 3, linear.output = TRUE, data=df)

plot(nn.3)

validation.preds.norm.nn.3 <- predict(nn.3, newdata=validation.norm)
accuracy(validation.norm$deaths_cases_ratio, validation.preds.norm.nn.3)

nn.1 <- neuralnet(deaths_cases_ratio ~ median_age	+ precent_65_and_over +	poverty_rate + percent_uninsured	+ percent_bachelors_or_higher + median_household_income
                  , train.norm, hidden = 1, linear.output = TRUE, data=df)

plot(nn.1)

validation.preds.norm.nn.1 <- predict(nn.1, newdata=validation.norm)
accuracy(validation.norm$deaths_cases_ratio, validation.preds.norm.nn.1)

ggplot() +
  geom_point(mapping = aes(x=validation.preds.norm.nn.2, y=validation.norm$deaths_cases_ratio))

hidden.vals <- rep(c(0,1,2,3,5,10,20,30,40,50,60,70,80,90,100),5)
val.rmse.results <- c()
train.rmse.results <- c()

set.seed(12345)
for (hidden.val in hidden.vals) {
  
  # train the model
  # limit steps with stepmax, to reduce runtime
  nn.current <- neuralnet(deaths_cases_ratio ~ median_age	+ precent_65_and_over +	poverty_rate + percent_uninsured	+ percent_bachelors_or_higher, train.norm, 
                          hidden = hidden.val, linear.output = TRUE, stepmax = 10^4)
  
  # obtain predictions on the validation set
  validation.preds.norm.nn.current <- predict(nn.current, newdata=validation.norm)
  train.preds.norm.nn.current <- predict(nn.current, newdata=train.norm)
  
  # measure the accuracy in terms of RMSE and save it
  val.rmse.current <- RMSE(validation.norm$deaths_cases_ratio, validation.preds.norm.nn.current)
  val.rmse.results <- c(val.rmse.results, val.rmse.current)
  
  train.rmse.current <- RMSE(train.norm$deaths_cases_ratio, train.preds.norm.nn.current)
  train.rmse.results <- c(train.rmse.results, train.rmse.current)
}

val.rmse.results
train.rmse.results <- c(train.rmse.results, train.rmse.current)


# training/test/validation split
set.seed(123456)
train.proportion <- 0.6
test.proportion <- 0.2
validation.proportion <- 0.2

train.index <- sample(1:nrow(), nrow(df)*train.proportion)
train.data <- df[train.index,]


holdout.data <- df[-train.index,]
test.index <- sample(1:nrow(holdout.data), nrow(df)*test.proportion)
test.data <- holdout.data[test.index,]
validation.data <- holdout.data[-test.index,]

dim(train.data)
dim(validation.data)
dim(test.data)

# set up dataframes to hold normalized values
train.data.norm <- train.data
validation.data.norm <- validation.data
test.data.norm <- test.data

head(train.data)
names(train.data)
selected.vars <- c("median_age","precent_65_and_over","poverty_rate","percent_uninsured", "percent_bachelors_or_higher", "median_household_income", "deaths_cases_ratio")

normalizer <- preProcess(train.data[,selected.vars],
                         method = c("center", "scale"))
# method = "range")
train.data.norm[,selected.vars] <- 
  predict(normalizer, train.data[,selected.vars])
validation.data.norm[,selected.vars] <- 
  predict(normalizer, validation.data[,selected.vars])
test.data.norm[,selected.vars] <- 
  predict(normalizer, test.data[,selected.vars])

df.knn.6.preds <- FNN::knn(train = train.data.norm[,selected.vars],
                             test = validation.data.norm[,selected.vars],
                             cl = train.data.norm$deaths_cases_ratio,
                             k=6)
head(df.knn.6.preds)
length(df.knn.6.preds)
summary(df.knn.6.preds)
summary(validation.data.norm$deaths_cases_ratio)


df.tibble <- as_tibble(df)
df.numeric <- df.tibble %>%
  select(median_age,precent_65_and_over,poverty_rate,percent_uninsured, percent_bachelors_or_higher, median_household_income, deaths_cases_ratio)

normalizer <- preProcess(df.numeric,
                         method = c("center", "scale"))

df.normalized <- predict(normalizer, df.numeric)

km.2 <- kmeans(df.normalized, 
               centers=2)

# look at a summary of the results
km.2

km.2$centers

kvals <- c(1,2,3,4,5,6,7,8,9,10)
k_tot_withinss <- c()
for (kval in kvals) {
  km_k <- kmeans(df.normalized, kval)
  current_tot_withinss <- km_k$tot.withinss
  k_tot_withinss <- c(k_tot_withinss, current_tot_withinss)
}

ggplot() +
  geom_line(mapping = aes(x=kvals, y=k_tot_withinss)) + 
  xlab("k") + ylab("Total Sum of Squares within Clusters")

km.6 <- kmeans(df.normalized, 
               centers=6)

# look at a summary of the results
km.6


table(df$deaths_cases_ratio)
df$ratio.status <- "Rejects"
df$ratio.status[df$deaths_cases_ratio] <- "Accepts"
df$ratio.status <- factor(df$ratio.status,levels=c("Accepts","Rejects"))

set.seed(123456)
train.proportion <- 0.6
test.proportion <- 0.2
validation.proportion <- 0.2

train.index <- sample(1:nrow(df), nrow(df)*train.proportion)
train.data <- df[train.index,]

holdout.data <- df[-train.index,]
test.index <- sample(1:nrow(holdout.data), nrow(df)*test.proportion)
test.data <- holdout.data[test.index,]
validation.data <- holdout.data[-test.index,]

dim(train.data)
dim(validation.data)
dim(test.data)

train.data.norm <- train.data
validation.data.norm <- validation.data
test.data.norm <- test.data

head(train.data)
names(train.data)
table(train.data$Education)
selected.vars <- c("median_age","precent_65_and_over","poverty_rate","percent_uninsured", "percent_bachelors_or_higher", "median_household_income", "deaths_cases_ratio")

normalizer <- preProcess(train.data[,selected.vars],
                         method = c("center", "scale"))
# method = "range")
train.data.norm[,selected.vars] <- 
  predict(normalizer, train.data[,selected.vars])
validation.data.norm[,selected.vars] <- 
  predict(normalizer, validation.data[,selected.vars])
test.data.norm[,selected.vars] <- 
  predict(normalizer, test.data[,selected.vars])
  
  
df.knn.6.preds <- FNN::knn(train = train.data.norm[,selected.vars],
                             test = validation.data.norm[,selected.vars],
                             cl = train.data.norm$deaths_cases_ratio,
                             k=6)

head(df.knn.6.preds)
length(df.knn.6.preds)
summary(df.knn.6.preds)
summary(validation.data.norm$ratio.status)

dim( attr(df.knn.6.preds, "nn.index") )
first.point.neighbor.indexes <- attr(df.knn.6.preds, "nn.index")[1,]
first.point.neighbor.indexes
first.point.neighbors <- train.data.norm[first.point.neighbor.indexes,selected.vars]

validation.data.norm[1,selected.vars]
first.point.neighbors

validation.data[1,selected.vars]
train.data[first.point.neighbor.indexes,selected.vars]

head( attr(df.knn.6.preds, "nn.dist") )
first.point.first.neighbor.distance <- attr(df.knn.6.preds, "nn.dist")[1,1]
first.point.first.neighbor.distance

validation.data.norm[1,selected.vars]
first.point.neighbors[1,selected.vars]

variable.distances <- validation.data.norm[1,selected.vars] - first.point.neighbors[1,selected.vars]
variable.distances
sqrt( sum(variable.distances^2) )
first.point.first.neighbor.distance

train.data.norm[first.point.neighbor.indexes,"ratio.status"]
df.status
















library(rpart) # for rpart function
library(rpart.plot)
library(gains)
library(caret) # for confusionMatrix
df <- read.csv("CensusFinal.csv")
head(df)

table(df$High)
df$Hi
gh <- ifelse(df$High,"Y","N")
df$High <- as.factor(df$High)
table(df$High)

set.seed(12345)
train.proportion <- 0.75
val.proportion <- 0.25

train.index <- sample(1:nrow(df), nrow(df)*train.proportion)
train.data <- df[train.index,]
validation.data <- df[-train.index,]


df.tree.1 <- rpart(High ~ median_age	+ precent_65_and_over +	poverty_rate + percent_uninsured	+ percent_bachelors_or_higher,
                     data=train.data,
                     method="class")

prp(df.tree.1, type=1, extra=1, under=TRUE, split.font=2, varlen=-10,
    main="Default")


cps <- c(0.1, 0.07, 0.05, 0.03, 0.02, 0.015,
         0.01, 0.007, 0.005, 0.003, 0.002, 0.0015,
         0.001, 0.0007, 0.0005, 0.0003, 0.0002, 0.00015, 0.0001)
balanced.accuracy.train <- c()
balanced.accuracy.val <- c()


for (cp in cps) {
  print(cp)
  df.tree.cp <- rpart(High ~ median_age	+ precent_65_and_over +	poverty_rate + percent_uninsured	+ percent_bachelors_or_higher,
                        data=train.data,
                        method="class",
                        cp=cp,
                        minsplit=2,
                        minbucket=1,
                        maxdepth=30,
                        xval=0
  )
  # get predictions for each value in the training and validation sets
  train.preds <- predict(df.tree.cp, newdata=train.data, type="class")
  val.preds <- predict(df.tree.cp, newdata=validation.data, type="class")
  
  # confusion matrix for train and validation sets
  cm.train <- confusionMatrix(train.preds, train.data$High)
  cm.val <- confusionMatrix(val.preds, validation.data$High)
  
  # pull out balanced accuracy from each confusion matrix and add it to the array
  balanced.accuracy.train <- c(balanced.accuracy.train,
                               cm.train$byClass['Balanced Accuracy'])
  balanced.accuracy.val <- c(balanced.accuracy.val,
                             cm.val$byClass['Balanced Accuracy'])
}

train.accuracy <- data.frame("cp"=cps,
                             "dataset"="Training",
                             "accuracy"=balanced.accuracy.train)
val.accuracy <- data.frame("cp"=cps,
                           "dataset"="Validation",
                           "accuracy"=balanced.accuracy.val)
full.accuracy <- rbind(train.accuracy, val.accuracy)
head(full.accuracy)

# take the inverse of Complexity Parameter 
# so that x axis shows increasing complexity
full.accuracy$cp.inverse <- 1 / full.accuracy$cp





set.seed(12345)
df.tree.xval <- rpart(High ~ median_age	+ precent_65_and_over +	poverty_rate + percent_uninsured	+ percent_bachelors_or_higher,
                        data=train.data,
                        method="class",
                        minsplit=2,
                        minbucket=1,
                        maxdepth=30,
                        xval=10,
                        cp=0.001)

# view the complexity parameter table
df.tree.xval$cptable

cp.df <- data.frame(df.tree.xval$cptable)
cp.df

# drop the first row for plotting as it's a big outlier
cp.df.plot <- cp.df[-1,]
# View validation performance vs CP for the rpart cptable
ggplot(data=cp.df.plot) + 
  geom_line(mapping = aes(x=CP, y=xerror)) + 
  xlab("Complexity Parameter") + 
  ylab("Cross-Validation Performance")
# scale_x_reverse() + 
# scale_y_reverse() + 
xlab("Complexity Parameter (reverse)") + 
  ylab("Cross-Validation Performance (1 - Error)")

best.cp.index <- which.min(df.tree.xval$cptable[,"xerror"])
best.cp.index
best.cp <- df.tree.xval$cptable[best.cp.index,"CP"]
best.cp

pruned <- prune(df.tree.xval, cp=best.cp)

prp(pruned, main="Pruned", fallen.leaves = FALSE, tweak=1.1)
summary(pruned)