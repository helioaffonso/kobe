import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, jaccard_similarity_score, accuracy_score


def mapCategories(categories):
    tMap = {}
    i = 0
    for value in categories:
        tMap[value] = i
        i = i+1
    return tMap

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

seasons = df['season'].str.slice(start=0,stop=4)
df['season_int'] = seasons.map(lambda x: int(x)-1996)

dfTest = df[df['shot_made_flag'].isnull()]
shotSeries = dfTest['shot_id']

df = df.dropna()

y = df['shot_made_flag'].values
X = df.drop(['team_name','shot_type', 'shot_zone_area', 'combined_shot_type', 'shot_zone_basic', 'shot_zone_range', 'matchup', 'opponent', 'action_type', 'team_id', 'season', 'shot_made_flag'], axis=1).values

testX = dfTest.drop(['team_name','shot_type', 'shot_zone_area', 'combined_shot_type', 'shot_zone_basic', 'shot_zone_range', 'matchup', 'opponent', 'action_type', 'team_id', 'season', 'shot_made_flag'], axis=1)

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(X,y)

predicted = forest.predict_proba(testX)

##dfPred = pd.DataFrame({'shot_id':shotSeries,'shot_made_flag':predicted}

#dfPredicted = pd.DataFrame({})

#dfPredicted['shot_id'] = shotSeries

#dfPredicted['shot_made_flag'] = predicted
#dfPredicted.to_csv('data/results.csv', sep=',')
#expected = testY

#print dfPredicted.head(20)


