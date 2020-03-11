
#This file goes over the functionality of the main prediction function of the main program. It uses multiple linear regression for the prediction.
def datasetPredict(csvF):
	
	#get our dataset that was given by calledbackDBPredict() and callBackDatasetPredict()
	dataset = csvF.copy()
	
	#Declare that we are going to use all the global varaiables. We store the results into them later.
	global database_correct
	global database_predictedvalues
	global database_bestcase
	global database_overall
	global database_Used 
	global database_module1
	global database_module2
	global database_module3
	global database_module4
	global database_coefficients1and2
	global database_coefficients3and4

	#If we predicted before, we need to drop the current results stored in the global variables.
	if database_Used == 1:
		database_overall = database_overall.drop([0],axis = 0)
		for index,row in database_module1.iterrows():
			database_module1 =  database_module1.drop([index], axis=0)
			database_module2 =  database_module2.drop([index], axis=0)
			database_module3 =  database_module3.drop([index], axis=0)
			database_module4 =  database_module4.drop([index], axis=0)
			database_bestcase = database_bestcase.drop([index], axis=0)
			database_correct = database_correct.drop([index], axis=0)
			database_predictedvalues = database_predictedvalues.drop([index], axis=0)
		for index,row in database_coefficients1and2.iterrows():
			database_coefficients1and2 = database_coefficients1and2.drop([index], axis=0)
		
		for index,row in database_coefficients3and4.iterrows():
			database_coefficients3and4 = database_coefficients3and4.drop([index], axis=0)


	pol = 0.0

	#Get the total number of movies of the given dataset.
	rowCount = 0
	rowCount = dataset.shape[0]
	
	#Create a number of control inputs equal to rowCount*9. These control values help in lessening the frequency of negative predicted values that are predicted.
	dataset.loc[rowCount] = ["Control"] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol] + 	[pol] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol]
	inet = 0
	inet = rowCount + 1
	while inet < rowCount*10:
		dataset.loc[inet]= ["Control"] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol] + 	[pol] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol] + [pol]
	
		inet = inet + 1
	

	#Create datasets that share the structure for datasets we need for every module prediction.
	#temp is used for testsets for modules 1 and 2, new is for the training dataset for modules 3 and 4, and temp2 is the testsets for modules 3 and 4.
	temp = pd.DataFrame(columns = ["MovieName", "PositiveTweets", 	"NeutralTweets","NegativeTweets",  "PositiveFavorites", "NeutralFavorites","NegativeFavorites", 	"PositiveRetweets", "NeutralRetweets","NegativeRetweets", "PositiveReplies", 	"NeutralReplies","NegativeReplies", "Budget", "BoxOfficeSales", "ProfitPercentage"])

	new = pd.DataFrame(columns = ["MovieName", "Positive", "Negative" , "Neutral", "Budget", "BoxOfficeSales", "ProfitPercentage"])

	temp2 = pd.DataFrame(columns = ["MovieName", "Positive", "Negative" , "Neutral", "Budget", "BoxOfficeSales", "ProfitPercentage"])
	inet = 0

	#Add the control values for the testset of new, that is used for modules 3 and 4.
	while inet < rowCount*10:
		new.loc[inet] = [dataset.at[inet, "MovieName"]] + [(dataset.at[inet, "PositiveTweets"] + dataset.at[inet, "PositiveFavorites"] + dataset.at[inet, "PositiveRetweets"] + dataset.at[inet, "PositiveReplies"])] + [(dataset.at[inet, "NegativeTweets"] + dataset.at[inet, "NegativeFavorites"] + dataset.at[inet, "NegativeRetweets"] + dataset.at[inet, "NegativeReplies"])] + [(dataset.at[inet, "NeutralTweets"] + dataset.at[inet, "NeutralFavorites"] + dataset.at[inet, "NeutralRetweets"] + dataset.at[inet, "NeutralReplies"])] +   [dataset.at[ inet, "Budget"]] +  [dataset.at[ inet, "BoxOfficeSales"]] + [dataset.at[ inet, "ProfitPercentage"]]
		#if inet < rowCount:
		#	print (new.loc[inet]) 
		inet = inet + 1
	#new.describe()
	

	#Create dataframes to keep track of  results.
	df = pd.DataFrame(columns = ["MovieName", "Actual", "Predicted"])
	predf = pd.DataFrame(columns = ["MovieName", "Actual", "Predicted"])

	#Keep track of the best value for each module.
	M2predf =  pd.DataFrame(columns = ["MovieName", "Actual", "Module 2 Prediction","Percentile Deviation", 'Best'])
	M1predf = pd.DataFrame(columns = ["MovieName", "Actual", "Module 1 Prediction", "Percentile Deviation",'Best'])
	M3predf =  pd.DataFrame(columns = ["MovieName", "Actual", "Module 3 Prediction", "Percentile Deviation", 'Best'])
	M4predf = pd.DataFrame(columns = ["MovieName", "Actual", "Module 4 Prediction",  "Percentile Deviation", 'Best'])

	#Keep track of all predicted values from each module and which one is the best.
	predicted_values =  pd.DataFrame(columns = ["MovieName", "Module 1", "M1 Ranking",  "Module 2", "M2 Ranking",  "Module 3", "M3 Ranking",  "Module 4", "M4 Ranking"])

	#Keep track of which module was the best for each movie.
	Best_modules =  pd.DataFrame(columns = ["MovieName", "Module 1" ,  "Module 2",  "Module 3",  "Module 4", "Best"])

	#Keep track of all the coefficients.
	Module12Coeff =  pd.DataFrame(columns = ["Module 1", "Module 2"])
	ModuleCoeff13 =  pd.DataFrame(columns = ['Coefficients',"Module 1", "Module 2"])
	Module34Coeff =  pd.DataFrame(columns = ["Module 3", "Module 4"])
	ModuleCoeff4 =  pd.DataFrame(columns = ['Coefficients',"Module 3", "Module 4"])



	iterations = 0
	
	#For every movie in our given dataset, do module 1
	while iterations < rowCount:
	
		#Add the movie whose row is equal to the iterations to the temporary dataset as the test dataset.
		temp.loc[0] = [dataset.at[ iterations, "MovieName"]] + [dataset.at[ iterations, "PositiveTweets"]] + [dataset.at[ iterations, "NeutralTweets"]] + [dataset.at[ iterations, "NegativeTweets"]] +  [dataset.at[ iterations, "PositiveFavorites"]] +  [dataset.at[ iterations, "NeutralFavorites"]] +   [dataset.at[ iterations, "NegativeFavorites"]] +  [dataset.at[ iterations, "PositiveRetweets"]] +   [dataset.at[ iterations, "NeutralRetweets"]] +  [dataset.at[ iterations, "NegativeRetweets"]] +  [dataset.at[ iterations, "PositiveReplies"]] +  [dataset.at[ iterations, "NeutralReplies"]]  +  [dataset.at[ iterations, "NegativeReplies"]] +  [dataset.at[ iterations, "Budget"]] +  [dataset.at[ iterations, "BoxOfficeSales"]] + [dataset.at[ iterations, "ProfitPercentage"]] 
		X_test = temp[["PositiveTweets", "NeutralTweets","NegativeTweets",  "PositiveFavorites", "NeutralFavorites","NegativeFavorites", "PositiveRetweets", "NeutralRetweets","NegativeRetweets", "PositiveReplies", "NeutralReplies","NegativeReplies", "Budget"]].values

		#Give the profitpercentage of the test movie.
		y_test = temp["ProfitPercentage"].values
		
		trainset = dataset

		#Drop the current movie from the trainingset, so we don't train our set with the testset.
		trainset = trainset.drop([iterations], axis=0)

		#Input the X train values and Y train values for the trainingset.
		X_train = trainset[["PositiveTweets", "NeutralTweets","NegativeTweets",  "PositiveFavorites", "NeutralFavorites","NegativeFavorites", "PositiveRetweets", "NeutralRetweets","NegativeRetweets", "PositiveReplies", "NeutralReplies","NegativeReplies", "Budget"]].values
		y_train = trainset['ProfitPercentage'].values

		#Create a linearregression model using sklearn library
		regressor = LinearRegression()
		regressor.fit(X_train, y_train)

		#Get the predicted value of of the testset compared to trainset results.
		y_pred = regressor.predict(X_test)

		#We only need to get the coeffiecients once, so get it for the second iteration.
		if(iterations == 1):		
			coeff_df = pd.DataFrame(regressor.coef_,  columns= ['Coefficients'])
			for index,row in coeff_df.iterrows():
				Module12Coeff.loc[index] = [coeff_df.at[index, 'Coefficients']] + [0]
				
		#Keep track of the current movie's actual result and predicted result.	
		df.loc[iterations] = [temp.at[0, "MovieName"]] + [temp.at[0, "ProfitPercentage"]] + [y_pred]
		#Keep track of the predicted values of the current movie for module 1.
		predicted_values.loc[iterations] =  [temp.at[0, "MovieName"]] +  [abs(y_pred*dataset.at[iterations, 'Budget'])] + [1] + [0] + [0] + [0] + [0] + [0] + [0]
	
		#Keep track of the module 1 results by themselves.
		M1predf.loc[iterations] =  [temp.at[0, "MovieName"]] + [temp.at[0, "BoxOfficeSales"]] + [abs(y_pred*dataset.at[iterations, 'Budget'])] + [0]+['']
		
		#If the result is negative, make it positive and divide by 3, to normalize it.
		if M1predf.at[iterations, 'Module 1 Prediction'] < 0:
			M1predf.at[iterations, 'Module 1 Prediction']= (abs(M1predf.at[iterations, 'Predicted'])/3)
		accuracy = 0

		#To show accurate percentile deviation values, if the value is less than 1, just times by 100, if it is above 1, but below 2, have 1 subtract the value that already(subtracted from 1) and then times by 100. Else, if the value is over 2, just subtract the value by 1 and times by 100
		accuracy = (M1predf.at[iterations, 'Module 1 Prediction']/M1predf.at[iterations, 'Actual'])
		if accuracy < 1 :
			accuracy = accuracy * 100.0
			M1predf.at[iterations, "Percentile Deviation"] = abs(100.0 - accuracy)
			
		elif accuracy < 2:
			accuracy =(1.0 - (accuracy - 1.0)) * 100.0
			M1predf.at[iterations, "Percentile Deviation"] = abs(100.0 - accuracy)
			
		else:
			accuracy = (accuracy - 1.0) * 100.0
			M1predf.at[iterations, "Percentile Deviation"] = accuracy
		
		iterations = iterations + 1
		temp = temp.drop([0], axis=0)
		
		
	iterations = 0
	print ("Done with module 1")

#So that is how a module is run. For next three sections, the exact same process is done, but now its for each module. No comments will be given, other than to indicate that a module section is done. This is due to the fact that the structure for 99% similar for them all. Modules 2 and 4 do have to convert Box Office Sales into values to match modules 1 and 3, and thats about it.


	while iterations < rowCount:
	
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
	
		if(iterations == 1):		
			coeff_df = pd.DataFrame(regressor.coef_,  columns= ['Coefficients'])
			for index,row in coeff_df.iterrows():
				Module12Coeff.at[index, 'Module 2'] = coeff_df.at[index, 'Coefficients']
		if (abs(df.at[iterations, 'Actual'] - df.at[iterations, 'Predicted'])) > (abs((y_pred/ temp.at[0, 'Budget']) - temp.at[0, 'ProfitPercentage'])):
		
			df.at[iterations, 'Predicted'] = (y_pred/ temp.at[0, 'Budget'])
			predicted_values.at[iterations, 'M2 Ranking'] = 1
			predicted_values.at[iterations, 'M1 Ranking'] = 0
			
	
		predicted_values.at[iterations, 'Module 2'] = y_pred
		M2predf.loc[iterations] =  [temp.at[0, "MovieName"]] + [temp.at[0, "BoxOfficeSales"]]+[abs(y_pred)] + [0] +['']
		if M2predf.at[iterations, 'Module 2 Prediction'] < 0:
			M2predf.at[iterations, 'Module 2 Prediction']= (abs(M2predf.at[iterations, 'Predicted'])/3)
		accuracy = 0
		accuracy = (M2predf.at[iterations, 'Module 2 Prediction']/M2predf.at[iterations, 'Actual'])
		if accuracy < 1 :
			accuracy = accuracy * 100.0
			M2predf.at[iterations, "Percentile Deviation"] = abs(100.0 - accuracy)
			
		elif accuracy < 2:
			accuracy =(1.0 - (accuracy - 1.0)) * 100.0
			M2predf.at[iterations, "Percentile Deviation"] = abs(100.0 - accuracy)
			
		else:
			accuracy = (accuracy - 1.0) * 100.0
			M2predf.at[iterations, "Percentile Deviation"] = accuracy
		iterations = iterations + 1
		temp = temp.drop([0], axis=0)
		
	print ("Done with module 2")

#The section for module 2 is now done.
	iterations = 0
	while iterations < rowCount:
	
		temp2.loc[0] = [new.at[ iterations, "MovieName"]] + [new.at[ iterations, "Positive"]] + [new.at[ iterations, "Negative"]] + [new.at[ iterations, "Neutral"]] +  [new.at[ iterations, "Budget"]] +  [new.at[ iterations, "BoxOfficeSales"]] + [new.at[ iterations, "ProfitPercentage"]] 
		X_test = temp2[["Positive", "Negative", "Neutral", "Budget"]].values
		y_test = temp2["ProfitPercentage"].values
		trainset = new
		trainset = trainset.drop([iterations], axis=0)
		X_train = trainset[["Positive", "Negative", "Neutral", "Budget"]].values
		y_train = trainset['ProfitPercentage'].values
		regressor = LinearRegression()
		regressor.fit(X_train, y_train)
		y_pred = regressor.predict(X_test)
		if(iterations == 1):		
			coeff_df = pd.DataFrame(regressor.coef_,  columns= ['Coefficients'])
			for index,row in coeff_df.iterrows():
				Module34Coeff.loc[index] = [coeff_df.at[index, 'Coefficients']] + [0]
		
		if (abs(df.at[iterations, 'Actual'] - df.at[iterations, 'Predicted'])) > (abs((y_pred) - temp2.at[0, 'ProfitPercentage'])):
		
			df.at[iterations, 'Predicted'] = (y_pred)
			predicted_values.at[iterations, 'M3 Ranking'] = 1
			predicted_values.at[iterations, 'M1 Ranking'] = 0
			predicted_values.at[iterations, 'M2 Ranking'] = 0
	
		predicted_values.at[iterations, 'Module 3'] = abs(y_pred * dataset.at[iterations, 'Budget']) 
	
		M3predf.loc[iterations] =  [temp2.at[0, "MovieName"]] + [temp2.at[0, "BoxOfficeSales"]] + [abs(y_pred*dataset.at[iterations, 'Budget'])]  + [0] + ['']
		if M3predf.at[iterations, 'Module 3 Prediction'] < 0:
			M3predf.at[iterations, 'Module 3 Prediction']= (abs(M3predf.at[iterations, 'Predicted'])/3)
		accuracy = 0
		accuracy = (M3predf.at[iterations, 'Module 3 Prediction']/M3predf.at[iterations, 'Actual'])
		if accuracy < 1 :
			accuracy = accuracy * 100.0
			M3predf.at[iterations, "Percentile Deviation"] = abs(100.0 - accuracy)
			
		elif accuracy < 2:
			accuracy =(1.0 - (accuracy - 1.0)) * 100.0
			M3predf.at[iterations, "Percentile Deviation"] = abs(100.0 - accuracy)
			
		else:
			accuracy = (accuracy - 1.0) * 100.0
			M3predf.at[iterations, "Percentile Deviation"] = accuracy
		iterations = iterations + 1
		temp2 = temp2.drop([0], axis=0)
		
	print ("Done with module 3")

#The section for module 3 is now done.
	iterations = 0
	while iterations < rowCount:
	
		temp2.loc[0] = [new.at[ iterations, "MovieName"]] + [new.at[ iterations, "Positive"]] + [new.at[ iterations, "Negative"]] + [new.at[ iterations, "Neutral"]] +  [new.at[ iterations, "Budget"]] +  [new.at[ iterations, "BoxOfficeSales"]] + [new.at[ iterations, "ProfitPercentage"]] 
		X_test = temp2[["Positive", "Negative", "Neutral", "Budget"]].values
		y_test = temp2["BoxOfficeSales"].values
		trainset = new
		trainset = trainset.drop([iterations], axis=0)
		X_train = trainset[["Positive", "Negative", "Neutral", "Budget"]].values
		y_train = trainset['BoxOfficeSales'].values
		regressor = LinearRegression()
		regressor.fit(X_train, y_train)
		y_pred = regressor.predict(X_test)
		if(iterations == 1):		
			coeff_df = pd.DataFrame(regressor.coef_,  columns= ['Coefficients'])
			for index,row in coeff_df.iterrows():
				Module34Coeff.at[index, 'Module 4'] = coeff_df.at[index, 'Coefficients']
		if (abs(df.at[iterations, 'Actual'] - df.at[iterations, 'Predicted'])) > (abs((y_pred/temp2.at[0, 'Budget']) - temp2.at[0, 'ProfitPercentage'])):
		
			df.at[iterations, 'Predicted'] = (y_pred/temp2.at[0, 'Budget'])
			predicted_values.at[iterations, 'M4 Ranking'] = 1
			predicted_values.at[iterations, 'M1 Ranking'] = 0
			predicted_values.at[iterations, 'M2 Ranking'] = 0
			predicted_values.at[iterations, 'M3 Ranking'] = 0

		predicted_values.at[iterations, 'Module 4'] = y_pred
	
		M4predf.loc[iterations] =  [temp2.at[0, "MovieName"]] + [temp2.at[0, "BoxOfficeSales"]]+[abs(y_pred)] + [0] + ['']
		if M4predf.at[iterations, 'Module 4 Prediction'] < 0:
			M4predf.at[iterations, 'Module 4 Prediction']= (abs(M4predf.at[iterations, 'Predicted'])/3)
		accuracy = 0
		accuracy = (M4predf.at[iterations, 'Module 4 Prediction']/M4predf.at[iterations, 'Actual'])
		if accuracy < 1 :
			accuracy = accuracy * 100.0
			M4predf.at[iterations, "Percentile Deviation"] = abs(100.0 - accuracy)
			
		elif accuracy < 2:
			accuracy =(1.0 - (accuracy - 1.0)) * 100.0
			M4predf.at[iterations, "Percentile Deviation"] = abs(100.0 - accuracy)
			
		else:
			accuracy = (accuracy - 1.0) * 100.0
			M4predf.at[iterations, "Percentile Deviation"] = accuracy
		iterations = iterations + 1
		temp2 = temp2.drop([0], axis=0)
		
	print ("Done with module 4")
	iterations = 0

#Module 4 is done, now we go get the best predicted value for each movie.
	accuracy = 0.0
	sum1 = 0.0
	df['Accuracy'] = 0.0

	#Get the percentile deviation of the best prediction.
	while iterations < rowCount:
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
		iterations = iterations + 1
		
	iterations = 0
	
	#For each movie, store its acutal and predicted values, and store them for the results and best case results that will be saved after this function is finished.
	while iterations < rowCount:
		df.at[iterations, 'Actual'] = dataset.at[iterations, 'BoxOfficeSales']
		df.at[iterations, 'Predicted']= (df.at[iterations, 'Predicted']*dataset.at[iterations, 'Budget'])
		database_predictedvalues.loc[iterations] = [predicted_values.at[iterations, 'MovieName']] + [predicted_values.at[iterations, 'Module 1']] + [predicted_values.at[iterations, 'Module 2']]  + [predicted_values.at[iterations, 'Module 3']]  + [predicted_values.at[iterations, 'Module 4']] + [''] + [''] + [0] + [0] + [0]


		database_bestcase.loc[iterations] = [df.at[iterations, "MovieName"]] + ['$' + ('%.2f' % (df.at[iterations, 'Actual']))] + ['$' + ('%.2f' % df.at[iterations, 'Predicted'])] + ['%.2f' % df.at[iterations, 'Accuracy']] + ['']
		iterations = iterations + 1
	iterations = 0 
	mean_act = []
	mean_pred = []

	#For each movie, append its actual value and predicted value for the variables for mean_absolute_error.
	while iterations < rowCount:
		
		mean_act.append(df.at[iterations, 'Actual'])
		mean_pred.append(df.at[iterations, 'Predicted'])
		iterations = iterations + 1
	
	#Save the overall results for the best case in 'database_overall'
	database_overall.loc[0] = [rowCount] + [0] + ['%' + ('%.2f' %(sum1/rowCount))] + [ '$' + ('%.2f' % (mean_absolute_error(mean_act,mean_pred)))] + [0] + [0]

	
	print("")
	
	print("")

	iterations = 0
	totalRig = 0


	#For each movie, we now need to guess what the best predicted value is.
	while iterations < rowCount:
	
		#Assume that the best ranking is module 2 initially for all movies.
		holdval = 'Module 2'
		holdnow = "M2 Ranking"


		#Keep track of the movies budget.
		Coldval = dataset.at[ iterations, "Budget"]

	
	
		#If a movie has $40,000,000 budget or more
		if (Coldval >= 40000000.0):
			store = 1
			
			#if a movie has less than $50,000,000 but more than $40 million, and total of all tweet related data is less than 12000, assign it module 3
			if ((Coldval <= 50000000.0 ) and (dataset.at[ iterations, "PositiveTweets"] + dataset.at[ iterations, "NeutralTweets"] + dataset.at[ iterations, "NegativeTweets"] +  dataset.at[ iterations, "PositiveFavorites"] +  dataset.at[ iterations, "NeutralFavorites"] +   dataset.at[ iterations, "NegativeFavorites"] +  dataset.at[ iterations, "PositiveRetweets"] +   dataset.at[ iterations, "NeutralRetweets"] +  dataset.at[ iterations, "NegativeRetweets"] +  dataset.at[ iterations, "PositiveReplies"] +  dataset.at[ iterations, "NeutralReplies"]  +  dataset.at[ iterations, "NegativeReplies"]) < 12000):
			
				holdval = 'Module 3'
				holdnow = 'M3 Ranking'
	

			#if a movie has less than $50,000,000 but more than $40 million, and total of all NegativeFavorites is less than 4000, assign it module 3
			if ((Coldval <= 50000000.0 ) and (dataset.at[ iterations, "NegativeFavorites"]) < 4000):
			
				holdval = 'Module 3'
				holdnow = 'M3 Ranking'
		
			
			
			
			
		#The movie has less than $40 million budget
		else:
			#Assume it has module 1 initially.
			holdval = 'Module 1'
			holdnow = 'M1 Ranking'



			#If the movie has less than $30 mil budget, and more than 100,000 tweet related interactions, make it module 2.
			if (dataset.at[ iterations, "PositiveTweets"] + dataset.at[ iterations, "NeutralTweets"] + dataset.at[ iterations, "NegativeTweets"] +  dataset.at[ iterations, "PositiveFavorites"] +  dataset.at[ iterations, "NeutralFavorites"] +   dataset.at[ iterations, "NegativeFavorites"] +  dataset.at[ iterations, "PositiveRetweets"] +   dataset.at[ iterations, "NeutralRetweets"] +  dataset.at[ iterations, "NegativeRetweets"] +  dataset.at[ iterations, "PositiveReplies"] +  dataset.at[ iterations, "NeutralReplies"]  +  dataset.at[ iterations, "NegativeReplies"]) > 100000 and Coldval < 30000000:
				holdval = 'Module 2'
				holdnow = 'M2 Ranking'

			
			#Else, if it has more than #35 budget and 100,000 tweet interactions, make it module 2.
			elif (dataset.at[ iterations, "PositiveTweets"] + dataset.at[ iterations, "NeutralTweets"] + dataset.at[ iterations, "NegativeTweets"] +  dataset.at[ iterations, "PositiveFavorites"] +  dataset.at[ iterations, "NeutralFavorites"] +   dataset.at[ iterations, "NegativeFavorites"] +  dataset.at[ iterations, "PositiveRetweets"] +   dataset.at[ iterations, "NeutralRetweets"] +  dataset.at[ iterations, "NegativeRetweets"] +  dataset.at[ iterations, "PositiveReplies"] +  dataset.at[ iterations, "NeutralReplies"]  +  dataset.at[ iterations, "NegativeReplies"]) > 100000 and Coldval > 35000000:
				holdval = 'Module 2'
				holdnow = 'M2 Ranking'
	
		#If a movie has less than 10,000 tweet interactions and lower than $20 million budget, assign it module 4.
		if (dataset.at[ iterations, "PositiveTweets"] + dataset.at[ iterations, "NeutralTweets"] + dataset.at[ iterations, "NegativeTweets"] +  dataset.at[ iterations, "PositiveFavorites"] +  dataset.at[ iterations, "NeutralFavorites"] +   dataset.at[ iterations, "NegativeFavorites"] +  dataset.at[ iterations, "PositiveRetweets"] +   dataset.at[ iterations, "NeutralRetweets"] +  dataset.at[ iterations, "NegativeRetweets"] +  dataset.at[ iterations, "PositiveReplies"] +  dataset.at[ iterations, "NeutralReplies"]  +  dataset.at[ iterations, "NegativeReplies"]) < 10000 and (Coldval <= 20000000):
			holdnow = 'M4 Ranking'
			holdval = 'Module 4'

		#Keep track of the total predictions that are correct. 
		if predicted_values.at[iterations, holdnow] == 1:
		
			totalRig = totalRig + 1

		#Store the predicted and best modules for each movie in best_modules. Check to see for each movie what module was best.
		for index,row in predicted_values.iterrows():
			if predicted_values.at[index, 'M1 Ranking'] == 1:
		
				Best_modules.loc[index] = [predicted_values.at[index, 'MovieName']]  + [predicted_values.at[index, 'Module 1']]  + [predicted_values.at[index, 'Module 2']] +  [predicted_values.at[index, 'Module 3']] +  [predicted_values.at[index, 'Module 4']] + ['Module 1']
			
			elif predicted_values.at[index, 'M2 Ranking'] == 1:
				
				Best_modules.loc[index] = [predicted_values.at[index, 'MovieName']]  + [predicted_values.at[index, 'Module 1']]  + [predicted_values.at[index, 'Module 2']] +  [predicted_values.at[index, 'Module 3']] +  [predicted_values.at[index, 'Module 4']] + ['Module 2']

			elif predicted_values.at[index, 'M3 Ranking'] == 1:

				Best_modules.loc[index] = [predicted_values.at[index, 'MovieName']]  + [predicted_values.at[index, 'Module 1']]  + [predicted_values.at[index, 'Module 2']] +  [predicted_values.at[index, 'Module 3']] +  [predicted_values.at[index, 'Module 4']] + ['Module 3']
	
			else :
				Best_modules.loc[index] = [predicted_values.at[index, 'MovieName']]  + [predicted_values.at[index, 'Module 1']]  + [predicted_values.at[index, 'Module 2']] +  [predicted_values.at[index, 'Module 3']] +  [predicted_values.at[index, 'Module 4']] + ['Module 4']

		
		#Keep track of which module was the best again, for a different dataset for each module.		
		if holdnow == 'M1 Ranking':
			M1predf.at[iterations, 'Best'] = 'yes'
			M2predf.at[iterations, 'Best'] = 'no'
			M3predf.at[iterations, 'Best'] = 'no'
			M4predf.at[iterations, 'Best'] = 'no'
			
		elif holdnow == 'M2 Ranking':
			M1predf.at[iterations, 'Best'] = 'no'
			M2predf.at[iterations, 'Best'] = 'yes'
			M3predf.at[iterations, 'Best'] = 'no'
			M4predf.at[iterations, 'Best'] = 'no'
			
		elif holdnow == 'M3 Ranking':
			
			M1predf.at[iterations, 'Best'] = 'no'
			M2predf.at[iterations, 'Best'] = 'no'
			M3predf.at[iterations, 'Best'] = 'yes'
			M4predf.at[iterations, 'Best'] = 'no'
		else:
			
			M1predf.at[iterations, 'Best'] = 'no'
			M2predf.at[iterations, 'Best'] = 'no'
			M3predf.at[iterations, 'Best'] = 'no'
			M4predf.at[iterations, 'Best'] = 'yes'
			


		#Keep track of the prediction value that the algorithm thopught was best.
		predf.loc[iterations]= [predicted_values.at[iterations, 'MovieName']] + [dataset.at[iterations, 'BoxOfficeSales']] + [predicted_values.at[iterations, holdval]]	
		
		#Keep track of what the predicted best module is.
		database_predictedvalues.at[iterations, 'Predicted Module'] = holdval
		
		iterations = iterations + 1
	

	
	iterations = 0
	accuracy = 0.0
	sum1 = 0.0
	predf['Accuracy'] = 0.0
	#For each movie, get the accuracy of the predicted best value is.
	while iterations < rowCount:
		if predf.at[iterations, 'Predicted'] < 0:
			predf.at[iterations, 'Predicted']= (abs(predf.at[iterations, 'Predicted'])/3)
		accuracy = (predf.at[iterations, 'Predicted']/predf.at[iterations, 'Actual'])
		if accuracy < 1 :
			accuracy = accuracy * 100.0
			predf.at[iterations, 'Accuracy'] = abs(100.0 - accuracy)
			sum1 = sum1 + (abs(100.0 - accuracy))
		elif accuracy < 2:
			accuracy =(1.0 - (accuracy - 1.0)) * 100.0
			predf.at[iterations, 'Accuracy'] = abs(100.0 - accuracy)
			sum1 = sum1 + (abs(100.0 - accuracy))
		else:
			accuracy = (accuracy - 1.0) * 100.0
			predf.at[iterations, 'Accuracy'] = accuracy
			sum1 = sum1 + accuracy
		iterations = iterations + 1
	
	iterations = 0
	mean_actual = []
	mean_predicted = []

	#Store all the results for either datasets that will be displayed, or for the result datasets that are global.
	while iterations < rowCount:
		predf.at[iterations, 'Actual'] = dataset.at[iterations, 'BoxOfficeSales']
		mean_actual.append(predf.at[iterations, 'Actual'])
		mean_predicted.append(predf.at[iterations, 'Predicted'])
		database_bestcase.at[iterations, 'Best module'] = Best_modules.at[iterations, 'Best']
		database_predictedvalues.at[iterations, 'Best'] = Best_modules.at[iterations, 'Best']
		database_predictedvalues.at[iterations, 'Predicted Value'] = predf.at[iterations, 'Predicted']
		database_predictedvalues.at[iterations, 'Percentile Deviation'] = predf.at[iterations, 'Accuracy']
		database_predictedvalues.at[iterations, 'Actual'] = predf.at[iterations, 'Actual']
		if database_predictedvalues.at[iterations, 'Best'] == database_predictedvalues.at[iterations, 'Predicted Module']:
			database_correct.loc[iterations]=[database_predictedvalues.at[iterations, 'MovieName']] + ['Yes']
		else:
			database_correct.loc[iterations]=[database_predictedvalues.at[iterations, 'MovieName']] + ['No']
		iterations = iterations + 1


	
	dt = predf.sort_values(['Accuracy'])

	
	nml = database_predictedvalues.copy()
	sml = nml.sort_values(['Percentile Deviation'])
	database_predictedvalues = sml

	#Add $ signs and keep all floats to 2 decimal places so that data is displayed better.
	for index,row in database_predictedvalues.iterrows():
		database_predictedvalues.at[index, 'Predicted Value'] = '$' + '%.2f' % database_predictedvalues.at[index, 'Predicted Value']
		database_predictedvalues.at[index, 'Actual'] = '$' + '%.2f' % database_predictedvalues.at[index, 'Actual']
		database_predictedvalues.at[index, 'Module 1'] = '$' + '%.2f' % database_predictedvalues.at[index, 'Module 1']
		database_predictedvalues.at[index, 'Module 2'] = '$' + '%.2f' % database_predictedvalues.at[index, 'Module 2']
		database_predictedvalues.at[index, 'Module 3'] = '$' + '%.2f' % database_predictedvalues.at[index, 'Module 3']
		database_predictedvalues.at[index, 'Module 4'] = '$' + '%.2f' % database_predictedvalues.at[index, 'Module 4']
		database_predictedvalues.at[index, 'Percentile Deviation'] = '%' + '%.2f' % database_predictedvalues.at[index, 'Percentile Deviation']
		dt.at[index, 'Predicted'] = '$' + '%.2f' % dt.at[index, 'Predicted']
		
		dt.at[index, 'Accuracy'] =   dt.at[index, 'Accuracy']
	display (database_predictedvalues)

	
	
	pd.set_option('display.max_rows',60)
	

	#Start printing out the results of the predicted best values, including how many were correctly chosen, the average percentile deviation, and the mean absolute error.
	print ("Total Best Modules Chosen:")
	print (totalRig, "/", rowCount)
	print ( '%.2f' % (sum1/rowCount), "= Mean Percentile Deviation")
	print('MAE', '$' + '%.2f' % (mean_absolute_error(mean_actual,mean_predicted)))
	database_overall.at[0,'Predicted MAE'] = '$' + '%.2f' % (mean_absolute_error(mean_actual,mean_predicted))
	database_overall.at[0,'Best Cases chosen'] = totalRig
	database_overall.at[0,'Predicted Percentile Deviation'] = '%' + '%.2f' % (sum1/rowCount)
	
	

		
	#Store the results of each module into the global datasets.
	database_module1 = M1predf.sort_values(['Best'],ascending=False)
	database_module2 = M2predf.sort_values(['Best'],ascending=False)
	database_module3 = M3predf.sort_values(['Best'],ascending=False)
	database_module4 = M4predf.sort_values(['Best'],ascending=False)
	
	#Indicate that we did a prediction, so that we know to drop global variable values if this function is run again.
	database_Used = 1

	#Get all the names for coefficients for the global coefficient datasets.
	columnames = dataset.columns.values.tolist()
	for index,row in Module12Coeff.iterrows():
				ModuleCoeff13.loc[index] = [columnames[index+1]] + [Module12Coeff.at[index, 'Module 1']] + [Module12Coeff.at[index, 'Module 2']] 
	columname = temp2.columns.values.tolist()
	for index,row in Module34Coeff.iterrows():
				ModuleCoeff4.loc[index] = [columname[index+1]] + [Module34Coeff.at[index, 'Module 3']] + [Module34Coeff.at[index, 'Module 4']] 
	

	#Save the coefficient values in the global datasets.
	database_coefficients1and2 = ModuleCoeff13
	database_coefficients3and4 = ModuleCoeff4
	
	#Save the best case modules by which ones were able to choose their best.
	bt = Best_modules.sort_values(['Best'])
	
	return
