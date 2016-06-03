import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, jaccard_similarity_score, accuracy_score
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def mapCategories(categories):
    tMap = {}
    i = 0
    for value in categories:
        tMap[value] = i
        i = i+1
    return tMap

def overtime(period):
	if period > 4:
		return 1
	else:
		return 0

def plot():
	alpha = 0.02
	plt.figure(figsize=(10,10))
	df = pd.read_csv('data/data.csv', sep=',', header=0).dropna()

	df['locx1'] = df['loc_x'].map(lambda x: abs(x))
	df['locy1'] = df['loc_y'].map(lambda x: abs(x))

	plt.subplot(121)
	plt.scatter(df.locx1, df.locy1, color='blue', alpha=alpha)
	plt.title('loc_x and loc_y')
	plt.show()

def getDataFrame():

	combShotTypeMap = {'Jump Shot':0, 'Dunk': 1, 'Layup':2, 'Tip Shot':3, 'Hook Shot':4, 'Bank Shot':5}
	shotZoneBasicMap = {'Mid-Range':0, 'Restricted Area':1, 'In The Paint (Non-RA)': 2, 'Above the Break 3':3, \
	                    'Right Corner 3':4, 'Backcourt': 5, 'Left Corner 3' : 6}
	shotZoneMap = {'Right Side(R)':0, 'Left Side(L)':1, 'Left Side Center(LC)':2, 'Right Side Center(RC)':3, 'Center(C)':4, 'Back Court(BC)':5}
	shotTypeMap = {'2PT Field Goal':2, '3PT Field Goal':3}
	shotZoneRangeMap = {'16-24 ft.':0, '8-16 ft.':1, 'Less Than 8 ft.':2, '24+ ft.':3, 'Back Court Shot':4}

	df = pd.read_csv('data/data.csv', sep=',', header=0)

	df['shot_type_alt'] = df['shot_type'].map(shotTypeMap)
	df['shot_zone_area_alt'] = df['shot_zone_area'].map(shotZoneMap)
	df['combined_shot_type_alt'] = df['combined_shot_type'].map(combShotTypeMap)
	df['shot_zone_basic_alt'] = df['shot_zone_basic'].map(shotZoneBasicMap)
	df['shot_zone_range_alt'] = df['shot_zone_range'].map(shotZoneRangeMap)
	df['opponent_alt'] = df['opponent'].map(mapCategories(df['opponent'].unique()))
	df['action_type_alt'] = df['action_type'].map(mapCategories(df['action_type'].unique()))
	df['game_date'] = pd.to_datetime(df['game_date'])
	df['game_date'] = df['game_date'].map(lambda x: x.toordinal())
	df['game_date'] = df['game_date'].map(lambda x: x - df.game_date.min())
	df['total_elapsed_time'] = ((df.period-1)*12*60) + ((11-df.minutes_remaining)*60) + (60 - df.seconds_remaining)
	df['remaning_time'] = df['minutes_remaining']*60+df['seconds_remaining']
	df['overtime'] = df['period'].map(lambda x: int(x>4))
	df['arc'] = np.arctan2(df['loc_x'],df['loc_y'])

	seasons = df['season'].str.slice(start=0,stop=4)
	df['season_int'] = seasons.map(lambda x: int(x)-1996)

	return df

plot()
#expected = testY
#print dfPredicted.head(20)


