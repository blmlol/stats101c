library(Matrix)
library(MASS)
library(gtools)
library(xgboost)
library(lubridate)

##Reading in the data
lafdtesting <- read.csv("C:/Users/Davey Wong/Desktop/testing.without.response.csv", stringsAsFactors=FALSE)
lafdtraining <- read.csv("C:/Users/Davey Wong/Desktop/lafdtraining(1).csv", stringsAsFactors=FALSE)


###Clean-up and Organization
lafdtraining <- na.omit(lafdtraining)
lafdtesting$isTest <- rep(1, nrow(lafdtesting)) ##I need to keep track of the testing data once I merge it with the training data
lafdtraining$isTest <- rep(0, nrow(lafdtraining)) ##I need to keep track of the testing data once I merge it with the testing data
  



##We noticed that there was some missing dispatch sequences so we replaced them with the median which was 1
summary(lafdtesting$Dispatch.Sequence)
lafdtesting$Dispatch.Sequence[is.na(lafdtesting$Dispatch.Sequence)] <- 1 


##Transformation
### Wanted to see if there was anyway I could transform the response variable
###We have to use a smaller subset since lm can't handle
trainer <- lafdtraining[sample(1:dim(lafdtraining)[[1]], 2000, replace = F),]
mfull = lm(elapsed_time~ Dispatch.Sequence + Unit.Type + PPE.Level +Incident.Creation.Time..GMT. + Dispatch.Status + year + First.in.District, data = trainer)
summary(mfull)
bcoxlafd <- boxcox(mfull)
lambda <- bcoxlafd$x[which.max(bcoxlafd$y)]
print(lambda)

#The max lambda value was closest to 0 which suggests a log transformation 
lafdtraining$elapsed_time <- log(lafdtraining$elapsed_time)




#### Combining the data using smartbind which allows for NA's when cbind-ing and rbind-ing 

fulldat <- smartbind(lafdtesting, lafdtraining) 
names(fulldat)
##We agreed that row id, incident id, year, and Emergency Dispatch Code were irrelevant predictors, so we remove them
fulldat <- fulldat[, c(-1,-2,-3, -5)]

fulldat$elapsed_time[is.na(fulldat$elapsed_time)] <- 0 #Set all the NAs in the testing data to 0 (response variable)


###We looked at the map of District locations and realized the the numbers were kind of arbitrary, but seemed to be a decent predictor, so I wanted to standardize it somehow
fulldat$First.in.District <- (fulldat$First.in.District - mean(fulldat$First.in.District))/sd(fulldat$First.in.District)



#There's no point in having a separate factor for every possible time instance so we extract just the hour and make that a factor
fulldat$Incident.Creation.Time..GMT. <- factor(hour(hms(fulldat$Incident.Creation.Time..GMT.)))

##Making the rest of predictors factors as well
fulldat$Dispatch.Status <- as.factor(fulldat$Dispatch.Status)
fulldat$Unit.Type <- as.factor(fulldat$Unit.Type)
fulldat$PPE.Level <- as.factor(fulldat$PPE.Level)


#####Splitting back into training and testing data
test <- fulldat[fulldat$isTest == 1,]
train <- fulldat[fulldat$isTest == 0,]
test <- test[,-7]
train <- train[,-7]

####Using XGBOOST


###Making sparse matrices
trainmat <- sparse.model.matrix(elapsed_time~.-1, data = train)
testmat <- sparse.model.matrix(elapsed_time~.-1, data = test)


####Running xgboost
xgblasttry <- xgboost(data = trainmat, label=train$elapsed_time, max_depth = 1000, eta = .3, nrounds = 100, early_stopping_rounds = 3)
cat('The best train-rmse was 0.408452')

####Making the Predictions
predtry <- predict(xgblasttry, testmat) 
predtry <- exp(predtry) #The predictions are of log values, so I need to exponentiate them

####Making sure we have reasonable results
summary(predtry)


####Submission!
predframe <- data.frame(row.id = lafdtesting$row.id, prediction = predtry) ##Creating a data frame with correct format
write.csv(predframe, file = 'pleasework3.csv', row.names = F) ##Writing the csv file
