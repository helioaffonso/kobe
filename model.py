import pandas as pd
import numpy as np
import transform
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, jaccard_similarity_score, accuracy_score, log_loss
from sklearn import cross_validation
from sklearn.cross_validation import KFold

import scipy as sp

def ensemble():
	df = transform.getDataFrame()
	dfTest = df[df['shot_made_flag'].isnull()]
	shotSeries = dfTest['shot_id']

	df = df.dropna()

	y = df['shot_made_flag'].as_matrix()
	X = df.drop(['game_event_id', 'lat', 'lon', 'team_name','shot_type', 'game_id', 'shot_zone_area', 'combined_shot_type', 'minutes_remaining', 'seconds_remaining', 'shot_zone_basic', 'shot_zone_range', 'matchup', 'opponent', 'action_type', 'team_id', 'season', 'shot_made_flag'], axis=1).as_matrix()

	kf = KFold(X.shape[0],n_folds=6, shuffle=True)

	#clfX = xgb.XGBClassifier(learning_rate=0.1, n_estimators=50,max_depth=5, min_child_weight=1, subsample=0.8,scale_pos_weight=1,colsample_bytree=0.8,gamma=0,seed=27)
	#clfX = xgb.XGBClassifier(learning_rate=0.1, n_estimators=50,max_depth=5, min_child_weight=1, subsample=0.8,scale_pos_weight=1,colsample_bytree=0.8,gamma=0)
	clfX = xgb.XGBClassifier(n_estimators=50,max_depth=5)
	clfRF = RandomForestClassifier(n_estimators=80)
	clfGB = GradientBoostingClassifier(n_estimators=80)

	probs = []
	probsXG = []
	probsRF = []
	probsGB = []

	for train_index, test_index in kf:
		#print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		clfX = clfX.fit(X_train,y_train)
		clfRF = clfRF.fit(X_train,y_train)
		clfGB = clfGB.fit(X_train,y_train)

		predictionX = clfX.predict_proba(X_test)
		predictionRF = clfRF.predict_proba(X_test)
		predictionGB = clfGB.predict_proba(X_test)

		dfPred = pd.DataFrame({})
		dfPred['XG'] = pd.Series(predictionX[:,1])
		dfPred['RF'] = pd.Series(predictionRF[:,1])
		dfPred['GB'] = pd.Series(predictionGB[:,1])

		dfPred['XGRF'] = (2*dfPred['XG'] + 2*dfPred['GB'] + dfPred['RF'])/5

		#print y_test.shape, type(y_test.shape)
		#print dfPred['XGRF'].shape, type(dfPred['XGRF'].shape)

		loss   = transform.logloss(y_test,dfPred['XGRF'].values)
		lossXG = transform.logloss(y_test,predictionX[:,1])
		lossRF = transform.logloss(y_test,predictionRF[:,1])
		lossGB = transform.logloss(y_test,predictionGB[:,1])

		probs.append(loss)
		probsXG.append(lossXG)
		probsRF.append(lossRF)
		probsGB.append(lossGB)

	print probs, np.array(probs).mean()
	print probsXG, np.array(probsXG).mean()
	print probsRF, np.array(probsRF).mean()
	print probsGB, np.array(probsGB).mean()


def KfoldItems():
	df = transform.getDataFrame()
	dfTest = df[df['shot_made_flag'].isnull()]
	shotSeries = dfTest['shot_id']

	df = df.dropna()

	y = df['shot_made_flag'].as_matrix()
	X = df.drop(['team_name','shot_type', 'game_id', 'shot_zone_area', 'combined_shot_type', 'shot_zone_basic', 'shot_zone_range', 'matchup', 'opponent', 'action_type', 'team_id', 'season', 'shot_made_flag'], axis=1).as_matrix()

	kf = KFold(X.shape[0],n_folds=10, shuffle=True)

	#clfX = xgb.XGBClassifier(learning_rate=0.1, n_estimators=50,max_depth=5, min_child_weight=1, subsample=0.8,scale_pos_weight=1,colsample_bytree=0.8,gamma=0,seed=27)
	#clfX = xgb.XGBClassifier(learning_rate=0.1, n_estimators=50,max_depth=5, min_child_weight=1, subsample=0.8,scale_pos_weight=1,colsample_bytree=0.8,gamma=0)
	#clfX = xgb.XGBClassifier(n_estimators=50,max_depth=5)
	clfX = GradientBoostingClassifier(n_estimators=250)

	probs = []

	for train_index, test_index in kf:
		print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		clfX = clfX.fit(X_train,y_train)
		prediction = clfX.predict_proba(X_test)

		print y_test.shape, type(y_test)
		print prediction[:,1].shape, type(prediction[:,1])

		loss = transform.logloss(y_test,prediction[:,1])
		probs.append(loss)

	print probs
	print np.array(probs).mean()



def crossValScore():
	df = transform.getDataFrame()
	dfTest = df[df['shot_made_flag'].isnull()]
	shotSeries = dfTest['shot_id']

	df = df.dropna()

	y = df['shot_made_flag'].as_matrix()
	X = df.drop(        ['team_name','shot_type', 'shot_zone_area', 'combined_shot_type', 'shot_zone_basic', 'shot_zone_range', 'matchup', 'opponent', 'action_type', 'team_id', 'season', 'shot_made_flag'], axis=1).as_matrix()

	testX = dfTest.drop(['team_name','shot_type', 'shot_zone_area', 'combined_shot_type', 'shot_zone_basic', 'shot_zone_range', 'matchup', 'opponent', 'action_type', 'team_id', 'season', 'shot_made_flag'], axis=1).as_matrix()

	clfX = xgb.XGBClassifier(n_estimators=100,max_depth=6)
	clfX = clfX.fit(X,y)

	print cross_val_score(clfX,X,y,scoring="log_loss",cv=8).mean()
	
	predicted = clfX.predict_proba(testX)

	dfPredicted = pd.DataFrame({})
	dfPredicted['shot_id'] = shotSeries
	dfPredicted['shot_made_flag'] = predicted[:,1]
	dfPredicted.to_csv('data/resultsX1.csv', sep=',', index=False)
	

def splitData():
	df = transform.getDataFrame()
	df = df.dropna()
	shotSeries = df['shot_id']
	dfTarget = df['shot_made_flag']
	df = df.drop(['team_name','shot_type', 'shot_zone_area', 'combined_shot_type', 'shot_zone_basic', 'shot_zone_range', 'matchup', 'opponent', 'action_type', 'team_id', 'season', 'shot_made_flag','shot_made_flag'],axis=1)

	X = df.as_matrix()
	y = dfTarget.as_matrix()
	
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
	#cross_val_score(forest,data,target,scoring="log_loss",cv=6).mean()

	return X_train, X_test, y_train, y_test

def XGBoost():
	df = transform.getDataFrame()
	df = df.dropna()
	df = df.drop(['team_name','shot_type', 'shot_zone_area', 'combined_shot_type', 'shot_zone_basic', 'shot_zone_range', 'matchup', 'opponent', 'action_type', 'team_id', 'season', 'shot_made_flag','shot_made_flag'],axis=1)
	
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)
	clfX = xgb.XGBClassifier(n_estimators=100,max_depth=6)
	clfX = clfX.fit(X_train,y_train)
	predicted = clfX.predict_proba(X_test)
	print log_loss(y_test, predicted)

def randomForest(X_train, X_test, y_train, y_test):
	print "Start RandomForestClassifier"
	forest = RandomForestClassifier(n_estimators = 100)
	forest = forest.fit(X_train,y_train)
	predicted = forest.predict_proba(X_test)

	print log_loss(y_test,predicted)

	"""
	print confusion_matrix(y_test, predicted)
	print jaccard_similarity_score(y_test, predicted)
	print accuracy_score(y_test, predicted)
	print(classification_report(expected, predicted, target_names=['dead', 'alive']))
	"""

def adaBoost(X_train, X_test, y_train, y_test):
	print "Start AdaBoost"
	
	clfAda = AdaBoostClassifier(n_estimators=100)
	clf = clf.fit(X_train, y_train)

	predicted = clf.predict_proba(X_test)
	
	loss = transform.logloss(y_test,prediction[:,1])
	print loss
	#scores = cross_val_score(clf, iris.data, iris.target)
	#scores.mean()   

def extraTrees(X_train, X_test, y_train, y_test):
	clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)


def submission():
	df = transform.getDataFrame()
	dfTest = df[df['shot_made_flag'].isnull()]
	shotSeries = dfTest['shot_id']

	df = df.dropna()

	y = df['shot_made_flag'].as_matrix()
	#X = df.drop(['team_name','shot_type', 'game_id', 'shot_zone_area', 'combined_shot_type', 'shot_zone_basic', 'shot_zone_range', 'matchup', 'opponent', 'action_type', 'team_id', 'season', 'shot_made_flag'], axis=1).as_matrix()
	X = df.drop(['game_event_id', 'lat', 'lon', 'team_name','shot_type', 'game_id', 'shot_zone_area', 'combined_shot_type', 'minutes_remaining', 'seconds_remaining', 'shot_zone_basic', 'shot_zone_range', 'matchup', 'opponent', 'action_type', 'team_id', 'season', 'shot_made_flag'], axis=1).as_matrix()

	testX = dfTest.drop(['game_event_id', 'lat', 'lon', 'team_name','shot_type', 'game_id', 'shot_zone_area', 'combined_shot_type', 'minutes_remaining', 'seconds_remaining', 'shot_zone_basic', 'shot_zone_range', 'matchup', 'opponent', 'action_type', 'team_id', 'season', 'shot_made_flag'], axis=1).values

	#clfX = xgb.XGBClassifier(learning_rate=0.1, n_estimators=50,max_depth=5, min_child_weight=1, subsample=0.8,scale_pos_weight=1,colsample_bytree=0.8,gamma=0,seed=27)
	clfRF = RandomForestClassifier(n_estimators=100)
	#clfX = xgb.XGBClassifier(learning_rate=0.1, n_estimators=50,max_depth=5, min_child_weight=1, subsample=0.8,scale_pos_weight=1,colsample_bytree=0.8,gamma=0)
	clfX = xgb.XGBClassifier(n_estimators=50,max_depth=5)	
	clfGB = GradientBoostingClassifier(n_estimators=80)

	clfX = clfX.fit(X,y)
	clfRF = clfRF.fit(X,y)
	clfGB = clfGB.fit(X,y)

	predictionX = clfX.predict_proba(testX)
	predictionRF = clfRF.predict_proba(testX)
	predictionGB = clfGB.predict_proba(testX)

	dfPred = pd.DataFrame({})
	dfPred['XG'] = pd.Series(predictionX[:,1])
	dfPred['RF'] = pd.Series(predictionRF[:,1])
	dfPred['GB'] = pd.Series(predictionGB[:,1])
	dfPred['XGRF'] = (2*dfPred['XG'] + 2*dfPred['GB'] + dfPred['RF'])/5

	dfPredicted = pd.DataFrame({})
	dfPredicted['shot_id'] = shotSeries
	dfPredicted['shot_made_flag'] = dfPred['XGRF'].values
	dfPredicted.to_csv('data/resultsXG-RF-GB-matchup2.csv', sep=',', index=False)


def standartModel():
	df = transform.getDataFrame()
	dfTest = df[df['shot_made_flag'].isnull()]
	shotSeries = dfTest['shot_id']

	df = df.dropna()

	y = df['shot_made_flag'].values
	X = df.drop(['team_name','shot_type', 'shot_zone_area', 'combined_shot_type', 'shot_zone_basic', 'shot_zone_range', 'matchup', 'opponent', 'action_type', 'team_id', 'season', 'shot_made_flag'], axis=1).values

	testX = dfTest.drop(['team_name','shot_type', 'shot_zone_area', 'combined_shot_type', 'shot_zone_basic', 'shot_zone_range', 'matchup', 'opponent', 'action_type', 'team_id', 'season', 'shot_made_flag'], axis=1)

	forest = RandomForestClassifier(n_estimators = 100)
	forest = forest.fit(X,y)

	predicted = forest.predict_proba(testX)

	dfPredicted = pd.DataFrame({})
	dfPredicted['shot_id'] = shotSeries
	dfPredicted['shot_made_flag'] = predicted[:,1]
	dfPredicted.to_csv('data/resultsX1.csv', sep=',', index=False)


#X_train, X_test, y_train, y_test = splitData()
#adaBoost(X_train, X_test, y_train, y_test)
#randomForest(X_train, X_test, y_train, y_test)

