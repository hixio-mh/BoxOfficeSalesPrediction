#This file goes over every global variable and function that is part of the main sales prediction program. This was done, as the original file feels to large to easily go through and read. Beyond that, there is only one main function "Dataset Predict", that is important, which is completely covered in "Dataset_Predict_Comments.py"


#These are all the dependencies, make sure to have all of these installed if you wish to use the program.
from tkinter import *
import pandas as pd
import numpy as np  
import os
import re 
import glob
import sys
import unicodedata
import math
import time
from textblob import TextBlob
import pymysql
import mysql.connector as mariadb
from sqlalchemy import create_engine


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from tkinter   import filedialog   
from tkinter.filedialog import askopenfilename


#These variables hold the font format used for buttons for the program.
LARGE_FONT= ("Verdana", 18)
Fine_font= ('Helvetica', 15)
small_font= ('Helvetica', 12)



#Connect to the sql database and read all entries from maindata to put into a dataset, tempDB.
db_connection = 'mysql+pymysql://account:pass@localhost/linear'
dbc= create_engine(db_connection)
mydb = pd.read_sql('select * from maindata', con=dbc)

tempDB = pd.DataFrame(columns = ["MovieName", "PositiveTweets", 	"NeutralTweets","NegativeTweets",  "PositiveFavorites", "NeutralFavorites","NegativeFavorites", 	"PositiveRetweets", "NeutralRetweets","NegativeRetweets", "PositiveReplies", 	"NeutralReplies","NegativeReplies", "Budget", "BoxOfficeSales", "ProfitPercentage"])

for index,row in mydb.iterrows():
	tempDB.loc[index] = [mydb.at[ index, "MovieName"]] + [mydb.at[ index, "PT"]] + [mydb.at[ index, "NT"]] + [mydb.at[ index, "NEGT"]] +  [mydb.at[ index, "PF"]] +  [mydb.at[ index, "NF"]] +   [mydb.at[ index, "NEGF"]] +  [mydb.at[ index, "PRT"]] +   [mydb.at[ index, "NRT"]] +  [mydb.at[ index, "NEGRT"]] +  [mydb.at[ index, "PRL"]] +  [mydb.at[ index, "NRL"]]  +  [mydb.at[ index, "NEGRL"]] +  [mydb.at[ index, "Budget"]] +  [mydb.at[ index, "BOX"]] + [mydb.at[ index, "PROFIT"]]

#Our defaultdataset that is used throughout the program becomes the initial dataset from the database.
defaultdataset = tempDB



#Our first group of global variables are here to store the results of predictions of a single movie that does or does not have box office. Open is for Box office, closed is for without. p_r_o_USed and p_r_c_Used is there to tell the system if a dataset is currently being stored for open and closed respectively.
ListofCSVS = {"firstCSV": '', "secondCSV": ''}

predicted_results_open = pd.DataFrame(columns = ["MovieName", "Module 1",  "Module 2",   "Module 3",   "Module 4", "Best Module", "Chosen Module", "Predicted Sales", "Actual Sales", "Percentile Deviation"])


p_r_o_Used = 0
predicted_results_close = pd.DataFrame(columns = ["MovieName", "Module 1",  "Module 2",   "Module 3",   "Module 4", "Chosen Module", "Predicted Sales"])


p_r_c_Used = 0



#Our second set of global datasets are here to store the results of a database or main dataset prediction. All 4 modules, the coefficients, all predicted values, the best case, all the best modules for each movie, and the over all results are all stored. 
database_module1 = pd.DataFrame(columns = ["MovieName", "Actual", "Module 1 Prediction", "Percentile Deviation", 'Best'])
database_module2 = pd.DataFrame(columns = ["MovieName", "Actual", "Module 2 Prediction", "Percentile Deviation",  'Best'])
database_module3 = pd.DataFrame(columns = ["MovieName", "Actual", "Module 3 Prediction", "Percentile Deviation",  'Best'])
database_module4 = pd.DataFrame(columns = ["MovieName", "Actual", "Module 4 Prediction", "Percentile Deviation",  'Best'])
database_coefficients1and2 =  pd.DataFrame(columns = ['Coefficients',"Module 1", "Module 2"])
database_coefficients3and4 =  pd.DataFrame(columns = ['Coefficients',"Module 3", "Module 4"])
database_predictedvalues =  pd.DataFrame(columns =["MovieName", "Module 1" ,  "Module 2",  "Module 3",  "Module 4",  "Predicted Module","Best", "Predicted Value", 'Actual', "Percentile Deviation" ])
database_correct = pd.DataFrame(columns =["MovieName", "Correct"])
database_bestcase = pd.DataFrame(columns = ["MovieName", "Actual", "Predicted", "Percentile Deviation", "Best module"])
database_overall =  pd.DataFrame(columns =["Total Movies", "Best Cases chosen", "Best Case Percentile Deviation", "Best Case MAE", "Predicted Percentile Deviation", "Predicted MAE"])

database_Used = 0 


#Function used to clean tweets for sentiment analysis for the Twitcalls() in BoxSentimentPage() and
# NoBoxSentimentPage(). 

def clean_tweet(tweet): 
  

#This comment applies a sentiment value from Textblob to a tweet given from the Twitcalls() in BoxSentimentPage() and NoBoxSentimentPage(). 
def get_tweet_sentiment(tweet): 


#Character parser to turn the characters of numbers into their numeric equivalent  for the Twitcalls() in BoxSentimentPage() and NoBoxSentimentPage()
def charparser(ch):


#Turn a string value for the total number of retweets and favorites of a tweet into a numerical value  for the Twitcalls() in BoxSentimentPage() and NoBoxSentimentPage()
def stringparser(st):


#The initial start up of the tkinter windows. All page layouts are defined here. Basically it keeps track of the windows that can appear in the program. 
class SeaofBTCapp(Tk):



#The start menu, that links to every other page and its function.
class StartPage(Frame):


#A page that lets the user see the structure of any CSV file, mainly for the use of seeing result files either with ViewResults() or ViewPercentileResults()     
class ViewResultPage(Frame):


#The function that runs in ViewResults() to show a csv file structure to a user. Mainly for results.
def ViewResults():


#Functiom that runs in ViewResults() that allows a user to see the percentile averages of a module file. 
def ViewPercentileResults():


#The page where a user can update the main database with callbackUpdateDB(): or replace the database with  callbackRewriteDB():
class UpdatePage(Frame):




#Function that takes a single Box.csv or BoxSentiment.csv file and adds it to the main database, while also updating defaultdataset. In UpdatePage()
def callbackUpdateDB():


#Function that deletes the current database and replaces it with a csv file. The program must restart after this is ran if the new database wishes to be used. In UpdatePage()
def callbackRewriteDB():


#The page where a user can predict for every movie in the database, a dataset, or to see the results. For prediction, datasetPredict() is used, and to see results is DatasetResultPage()
class DatasetPredictPage(Frame):

#Function to exit the program and windows for the none jupyter notebook version.
def exi(sel):

#The page that allows a user to choose whether they want to apply sentiment to a dataset and give the box office, or to a dataset and don't give box office. BoxSentimentPage() and NoBoxSentimentPage() are the linked pages.
class SentimentPage(Frame):

#Page that has 3 text fields for a user to input the Name of the movie, the budget of the movie, and the box office a movie. Once done, the inside Twitcall() function is called, which takes the chosen twitter dataset and applies sentiment to it and then counts the totals for all relevant values.
class BoxSentimentPage(Frame):


#The same as BoxSentimentPage(), just now only two text fields as box office is no longer inputted. The functionality is the same.
class NoBoxSentimentPage(Frame):

#The page that allows a user to combine two datasets of the same shape by choosing them and then uses the Comcall() inner function to combine them.
class CombinePage(Frame):


#The page that connects to DatasetPredictPage() to show all the results from the previous dataset prediction. Uses SaveDatasetResults() to save all results to a file.
class DatasetResultPage(Frame):

#A function that takes all the global result variables from the second group (the database_thing datasets) and creates a timestamped folder and then saves each one to that folder. Called from DatasetResultPage()
def SaveDatasetResults():

#Show the results of 'database_correct', part of DatasetResultPage()
def CorrectResult():


#Show the results of 'database_module1', part of DatasetResultPage()
def Module1Result():


#Show the results of 'database_module2', part of DatasetResultPage()
def Module2Result():

#Show the results of 'database_module3', part of DatasetResultPage()
def Module3Result():


#Show the results of 'database_module4', part of DatasetResultPage()
def Module4Result():
	
#Show the coefficients of 'database_coefficients1and2', part of DatasetResultPage()
def Coefficient12Result():


#Show the coefficients of 'database_coefficients3and4', part of DatasetResultPage()
def Coefficient34Result():
	
#Show the bestcase from 'database_bestcase', part of DatasetResultPage()
def BestCaseResult():
	
#Show the overall results of 'database_overall', part of DatasetResultPage()
def OverallResult():

#Show the coefficients of 'database_predictedvalues', part of DatasetResultPage()
def AllValuesResult():

#The initial function called by  DatasetPredictPage(), that selects the chosen csv file dataset as the set to be predicted by datasetPredict()	
def callbackDatasetPred():
	
#The initial function called by  DatasetPredictPage(), that has the current database set in defaultdataset be predicted by datasetPredict()
def callbackDBPred():


#The prediction function of the entire program. A more in depth explanation of it is in the 'Dataset_Predict_comments.py" file. All global 'database_name' variables are used and obtain values here. Both  callbackDBPred(): and  callbackDatasetPred(), send files to it.
def datasetPredict(csvF):
	
#The function used by CombinePage() to store the files chosen by user that will be combined.
def getCSV(name):

#The page where a user can predict the box office of a single movie against the main database set in defaultdataset. The movie can either have box office values, or not have box office values. A user can then see the results and save them. OpenResult(), ClosedResult(), SaveOpenResult(), SaveClosedResult(), OpenBoxPrediction(), and ClosedBoxPrediction() are all called from here.
class PredictPage(Frame):


#The function called by PredictPage(), to display the results of the last OpenBoxPrediction done stored in predicted_results_open. 
def OpenResult():


#The function called by  PredictPage(), to display the results of the last ClosedBoxPrediction done, stored in predicted_results_close.
def ClosedResult():


#The function called by  PredictPage(), to save the results of the last OpenBoxPrediction done, stored in predicted_results_open.
def SaveOpenResult():

	
#The function called by  PredictPage(), to save the results of the last ClosedBoxPrediction done, stored in predicted_results_close.
def SaveClosedResult():
	
	
	
	
#The initial function called by   PredictPage(), so that thee user can choose a single box office file for prediction for OpenBoxPrediction()
def callbackPredict():
	
#The initial function called by   PredictPage(), so that the user can choose a single no box office file for prediction for ClosedBoxPrediction()
def callbackPredict2():
	
#The functionality of this program is similar as datasetPredict(), so refer to "Dataset_Predict_comments.py" file if you wish to understand the structure. The only difference is that only the chosen file is predicted, and the main defaultdataset is used for training. p_r_o_Used and predicted_results_open obtain their values here.
def OpenBoxPrediction(csvP):

#Similar to OpenBoxPrediction(), the only difference being that the final predicted value is not compared to its actual box office, as that data was not given. p_r_c_Used and predicted_results_closed obtain their values here.	
def ClosedBoxPrediction(csvP):



#What lets the program run and loop.
app = SeaofBTCapp()
app.mainloop()



















