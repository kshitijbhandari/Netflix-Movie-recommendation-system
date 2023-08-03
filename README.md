# Netflix-Movie-recommendation-system

Problem Statement 

Netflix provided a lot of anonymous rating data, and a prediction accuracy bar that is 10% better than what Cinematch can do on the same training data set. (Accuracy is a measurement of how closely predicted ratings of movies match subsequent actual ratings.)

SOURCES :

      https://www.kaggle.com/netflix-inc/netflix-prize-data

      Netflix blog: https://medium.com/netflix-techblog/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429 

BUSINESS OBJECTIVE :

      Predict the rating that a user would give to a movie that he ahs not yet rated.
      Minimize the difference between predicted and actual rating (RMSE and MAPE) 

DATA :

      The first line of each file [combined_data_1.txt, combined_data_2.txt, combined_data_3.txt, combined_data_4.txt] contains the movie id followed by a colon. Each subsequent line in the file corresponds to a rating from a customer and its date in the following format:

      CustomerID,Rating,Date
