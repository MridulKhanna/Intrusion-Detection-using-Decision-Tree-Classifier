# Intrusion-Detection-using-Decision-Tree-Classifier

  Scikit-learn library (machine learning library) in Python has been used for detection of intrusion.Pandas library is used for reading the .csv files.
  
  
  Given a networking dataset(KDDCUP1999) the goal is to predict whether the feature corresponds to a normal feature or it is a smurf.
  The dataset consists of 41 features such as protocol type,service,src_bytes,dest_bytes,root_shell,num_root.
  After predicting the result or classifying the input feature,a decision tree is generated as dt.dot file which can be opened using XDot.
  
  
  Some of the functions used in Scikit-learn are:-
  1.fit(x,y) - Used to train the model from the given dataset
  2.predict(test data) - Used to classify the new incoming data to a specific target or a class
