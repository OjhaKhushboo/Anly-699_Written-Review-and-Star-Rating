# Load the dataset
ProductReviews <- read.csv("C:/MS Analytics/ANLY 699/New Project 6june/GrammarandProductReviews.csv")

#Perform Explorartory daa analysis
summary(ProductReviews)
str(ProductReviews)

#Total brands, categories
levels(ProductReviews$categories)
levels(ProductReviews$brand)
levels(ProductReviews$manufacturer)

#Frequency of product categories and brand
library(plyr)
count <- ddply(ProductReviews, .(ProductReviews$categories, ProductReviews$brand), nrow)
names(count) <- c("Product Categories", "Brand", "Freq")
count

#Range of star rating
hist(ProductReviews$reviews.rating, 
     main="Star Rating",
     xlab="Star")

#Word Cloud to see the occureence of nost common word used in the comment
library("NLP")
library("tm")
library("SnowballC")
library("RColorBrewer")
library("wordcloud")
text<-sample(ProductReviews$reviews.text,5000)
#Load the data as corpus. 
docs<-Corpus(VectorSource(text))
inspect(docs)
#Replacing URL and special charaters
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
docs<-tm_map(docs,content_transformer(toSpace))

r<-function(x)gsub("[^[:alpha:][:space:]]*","",x)
docs<-tm_map(docs,content_transformer(r))
# Remove numbers
docs <- tm_map(docs, removeNumbers)
# Remove english common stopwords
docs <- tm_map(docs, removeWords, stopwords("english"))
# Convert the text to lower case
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeWords,c("wanna","gonna","what","will","use"))
docs <- tm_map(docs, toSpace, "/")
docs <- tm_map(docs, toSpace, "@")
docs <- tm_map(docs, toSpace, "\\|")
# Remove punctuations
docs <- tm_map(docs, removePunctuation)
# Eliminate extra white spaces
docs <- tm_map(docs, stripWhitespace)
#Text stemming
#docs <- tm_map(docs, stemDocument)
#Build a term-document matrix
dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)

#Generate the word cloud
set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 10,
          max.words=Inf, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

#Sentiment analysis
library(syuzhet)
library(ggplot2)
senti<-get_nrc_sentiment(as.character(docs))
senti
#Rearrange the dataframe
Newdf<-data.frame(colSums(senti[,]))
names(Newdf)<-"Numbers"
Newdf<-cbind("sentiment"=rownames(Newdf),Newdf)
rownames(Newdf)<-NULL

#create Barplot using ggplot library
ggplot(data=Newdf,aes(x<-sentiment,y=Numbers))+
  geom_bar(aes(fill=sentiment),stat="identity")+
  theme(legend.position="none")+
  xlab("sentiment")+ggtitle("Sentiment Related to Reviews")

#Does the reviewer recommend the product
library(ggplot2)
ggplot(ProductReviews, aes(x=reviews.rating,fill=reviews.doRecommend)) +
  geom_bar()

#Did the reviewer buy the product
ggplot(ProductReviews, aes(x=reviews.rating, fill=reviews.didPurchase)) +   
  geom_bar()

plot(ProductReviews$reviews.numHelpful,type="l", ylab="No:of Helpful reviews", main="Number of helpful comments")


ggplot(ProductReviews, aes(x=reviews.rating,y=reviews.numHelpful,shape=reviews.didPurchase,
                           color=reviews.didPurchase)) + geom_point() +
  ylab('No:of helpful')+xlab('Rating')


# Building a model using logistic algorithem to predict if the highest star rating is more helpful. 

Hypo1<- ProductReviews[,17:18]
Hypo1
sum(is.na(Hypo1$Is_It_Helpful))
sum(is.na(Hypo1$reviews.numHelpful))
sum(is.na(Hypo1$reviews.rating))
Hypo1<-na.omit(Hypo1)
Hypo1$Is_It_Helpful<-NA
Hypo1 <- within(Hypo1, Is_It_Helpful[Hypo1$reviews.numHelpful>=1] <- 1)
Hypo1 <- within(Hypo1, Is_It_Helpful[Hypo1$reviews.numHelpful==0] <- 0)
Hypo1$Is_It_Helpful<-factor(Hypo1$Is_It_Helpful)


#Split the data
dt = sort(sample(nrow(Hypo1), nrow(Hypo1)*.7))
train<-Hypo1[dt,]
test<-Hypo1[-dt,]

model <- glm(Is_It_Helpful ~reviews.rating,family=binomial(link='logit'),data=train)
summary(model)
anova(model, test="Chisq")
fitted.results <- predict(model,newdata=subset(test,select=c(2,3)),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results != test$Is_It_Helpful)
print(paste('Accuracy',1-misClasificError))

library(ROCR)
p <- predict(model, newdata=subset(test,select=c(2,3)), type="response")
pr <- prediction(p, test$Is_It_Helpful)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

#Building the SVM model 
library("lattice")
library("ggplot2")
library("caret")

Hypo2<- ProductReviews[,c(17,18,20)]
Hypo2
#Length of word
Hypo2$WordCount<-"NA"
wordcount <- function(str) {
  sapply(gregexpr("\\b\\W+\\b", str, perl=TRUE), function(x) sum(x>0) ) + 1 
}

Hypo2$WordCount<-wordcount(Hypo2$reviews.text)

#How to decide if a review is short or long
#mean of the WordCount is used to find is the comment/text review is shrt or long
mean(Hypo2$WordCount)

Hypo2$Text.type<-"NA"
Hypo2 <- within(Hypo2, Text.type[Hypo2$WordCount>=mean(Hypo2$WordCount)] <-"Long review text")
Hypo2 <- within(Hypo2, Text.type[Hypo2$WordCount<mean(Hypo2$WordCount)] <- "Short review text")

#Is it helpfull
Hypo2<-na.omit(Hypo2)
Hypo2$Is_It_Helpful<-NA
Hypo2 <- within(Hypo2, Is_It_Helpful[Hypo2$reviews.numHelpful>=1] <- 1)
Hypo2 <- within(Hypo2, Is_It_Helpful[Hypo2$reviews.numHelpful==0] <- 0)
Hypo2$Is_It_Helpful<-factor(Hypo2$Is_It_Helpful)
is.factor(Hypo2$Is_It_Helpful)
str(Hypo2)

#Split the data
#The caret package provides a method createDataPartition() for partitioning our data into train and test set.

#We will create a sample dataframe(1500) to implement SVM model as we got error that system can't allocate vector of that size

set.seed(1234)
dt <- createDataPartition(y = Hypo2$Is_It_Helpful, p= 0.7, list = FALSE)
train<-Hypo2[dt,]
test<-Hypo2[-dt,]
dim(train); dim(test);

#Pre process
anyNA(Hypo2)
summary(Hypo2)
is.factor(Hypo2$Is_It_Helpful)


trctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)
set.seed(2222)

svm_Linear <- train(Is_It_Helpful~WordCount+reviews.rating, data = train, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 5)

svm_Linear
#Test
test_pred <- predict(svm_Linear, newdata = test)
test_pred
#Find how accurate our model is working
confusionMatrix(test_pred, test$Is_It_Helpful )


#Model 3 to 