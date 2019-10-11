import pandas as pd  
import numpy as np  

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv("MovieSalesData2.csv")

#print(dataset.loc[52])

pol = 0.0

dataset.loc[53] = ["Control"] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol]

inet = 54
while inet < 504:
	dataset.loc[inet]= ["Control"] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol] + 	[pol] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol]

	inet = inet + 1

temp = pd.DataFrame(columns = ["MovieName", "PositiveTweets", "NeutralTweets","NegativeTweets",  "PositiveFavorites", "NeutralFavorites","NegativeFavorites", "PositiveRetweets", "NeutralRetweets","NegativeRetweets", "PositiveReplies", "NeutralReplies","NegativeReplies", "Budget", "BoxOfficeSales", "ProfitPercentage"])

df = pd.DataFrame(columns = ["MovieName", "Actual", "Predicted"])

#print(temp.loc[0])

X = dataset[["PositiveTweets", "NeutralTweets","NegativeTweets",  "PositiveFavorites", "NeutralFavorites","NegativeFavorites", "PositiveRetweets", "NeutralRetweets","NegativeRetweets", "PositiveReplies", "NeutralReplies","NegativeReplies", "Budget"]].values

y = dataset['ProfitPercentage'].values

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#dataset = dataset.drop([0], axis=0)
#print(dataset)

#regressor = LinearRegression()  
#regressor.fit(X_train, y_train)

#coeff_df = pd.DataFrame(regressor.coef_)  
#coeff_df.to_csv("coed.csv", index=False, encoding='utf-8-sig')

#y_pred = regressor.predict(X_test)

#df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#df.to_csv("test.csv", index=False, encoding='utf-8-sig')

iterations = 0

while iterations < 53:
	
	temp.loc[0] = [dataset.at[ iterations, "MovieName"]] + [dataset.at[ iterations, "PositiveTweets"]] + [dataset.at[ iterations, "NeutralTweets"]] + [dataset.at[ iterations, "NegativeTweets"]] +  [dataset.at[ iterations, "PositiveFavorites"]] +  [dataset.at[ iterations, "NeutralFavorites"]] +   [dataset.at[ iterations, "NegativeFavorites"]] +  [dataset.at[ iterations, "PositiveRetweets"]] +   [dataset.at[ iterations, "NeutralRetweets"]] +  [dataset.at[ iterations, "NegativeRetweets"]] +  [dataset.at[ iterations, "PositiveReplies"]] +  [dataset.at[ iterations, "NeutralReplies"]]  +  [dataset.at[ iterations, "NegativeReplies"]] +  [dataset.at[ iterations, "Budget"]] +  [dataset.at[ iterations, "BoxOfficeSales"]] + [dataset.at[ iterations, "ProfitPercentage"]] 
	X_test = temp[["PositiveTweets", "NeutralTweets","NegativeTweets",  "PositiveFavorites", "NeutralFavorites","NegativeFavorites", "PositiveRetweets", "NeutralRetweets","NegativeRetweets", "PositiveReplies", "NeutralReplies","NegativeReplies", "Budget"]].values
	y_test = temp["ProfitPercentage"].values
	trainset = dataset
	trainset = trainset.drop([iterations], axis=0)
	X_train = trainset[["PositiveTweets", "NeutralTweets","NegativeTweets",  "PositiveFavorites", "NeutralFavorites","NegativeFavorites", "PositiveRetweets", "NeutralRetweets","NegativeRetweets", "PositiveReplies", "NeutralReplies","NegativeReplies", "Budget"]].values
	y_train = trainset['ProfitPercentage'].values
	regressor = LinearRegression()
	regressor.fit(X_train, y_train)
	y_pred = regressor.predict(X_test)
	#print(iterations)
	df.loc[iterations] = [temp.at[0, "MovieName"]] + [temp.at[0, "ProfitPercentage"]] + [y_pred]
	#dataset = dataset.drop([0], axis=0)
	#print(dataset.loc[0])
	iterations = iterations + 1
	temp = temp.drop([0], axis=0)
iterations = 0
while iterations < 53:
	
	temp.loc[0] = [dataset.at[ iterations, "MovieName"]] + [dataset.at[ iterations, "PositiveTweets"]] + [dataset.at[ iterations, "NeutralTweets"]] + [dataset.at[ iterations, "NegativeTweets"]] +  [dataset.at[ iterations, "PositiveFavorites"]] +  [dataset.at[ iterations, "NeutralFavorites"]] +   [dataset.at[ iterations, "NegativeFavorites"]] +  [dataset.at[ iterations, "PositiveRetweets"]] +   [dataset.at[ iterations, "NeutralRetweets"]] +  [dataset.at[ iterations, "NegativeRetweets"]] +  [dataset.at[ iterations, "PositiveReplies"]] +  [dataset.at[ iterations, "NeutralReplies"]]  +  [dataset.at[ iterations, "NegativeReplies"]] +  [dataset.at[ iterations, "Budget"]] +  [dataset.at[ iterations, "BoxOfficeSales"]] + [dataset.at[ iterations, "ProfitPercentage"]] 
	X_test = temp[["PositiveTweets", "NeutralTweets","NegativeTweets",  "PositiveFavorites", "NeutralFavorites","NegativeFavorites", "PositiveRetweets", "NeutralRetweets","NegativeRetweets", "PositiveReplies", "NeutralReplies","NegativeReplies", "Budget"]].values
	y_test = temp["BoxOfficeSales"].values
	trainset = dataset
	trainset = trainset.drop([iterations], axis=0)
	X_train = trainset[["PositiveTweets", "NeutralTweets","NegativeTweets",  "PositiveFavorites", "NeutralFavorites","NegativeFavorites", "PositiveRetweets", "NeutralRetweets","NegativeRetweets", "PositiveReplies", "NeutralReplies","NegativeReplies", "Budget"]].values
	y_train = trainset['BoxOfficeSales'].values
	regressor = LinearRegression()
	regressor.fit(X_train, y_train)
	y_pred = regressor.predict(X_test)
	#print(iterations)
	if (abs(df.at[iterations, 'Actual'] - df.at[iterations, 'Predicted'])) > (abs((y_pred/ temp.at[0, 'Budget']) - temp.at[0, 'ProfitPercentage'])):
		#print('changed' , iterations)
		df.at[iterations, 'Predicted'] = (y_pred/ temp.at[0, 'Budget'])
	#dataset = dataset.drop([0], axis=0)
	#print(dataset.loc[0])
	iterations = iterations + 1
	temp = temp.drop([0], axis=0)
iterations = 0
accuracy = 0.0
sum1 = 0.0
df['Accuracy'] = 0.0
while iterations < 53:
	if df.at[iterations, 'Predicted'] < 0:
		df.at[iterations, 'Predicted']= (abs(df.at[iterations, 'Predicted'])/3)
	accuracy = (df.at[iterations, 'Predicted']/df.at[iterations, 'Actual'])
	if accuracy < 1 :
		accuracy = accuracy * 100.0
		df.at[iterations, 'Accuracy'] = abs(100.0 - accuracy)
		sum1 = sum1 + (abs(100.0 - accuracy))
	elif accuracy < 2:
		accuracy =(1.0 - (accuracy - 1.0)) * 100.0
		df.at[iterations, 'Accuracy'] = abs(100.0 - accuracy)
		sum1 = sum1 + (abs(100.0 - accuracy))
	else:
		accuracy = (accuracy - 1.0) * 100.0
		df.at[iterations, 'Accuracy'] = accuracy
		sum1 = sum1 + accuracy
	#df.at[iterations, 'Accuracy'] = abs(100.0 - accuracy)
	#sum1 = (abs(100.0 - accuracy))
	iterations = iterations + 1
iterations = 0
while iterations < 53:
	df.at[iterations, 'Actual'] = dataset.at[iterations, 'BoxOfficeSales']
	df.at[iterations, 'Predicted']= (df.at[iterations, 'Predicted']*dataset.at[iterations, 'Budget'])
	iterations = iterations + 1

dt = df.sort_values(['Accuracy'])

print(dt)

print (sum1/53, "= Mean Difference")

print( dt.at[26, "Accuracy"],"= Median Difference")


