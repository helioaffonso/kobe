import pandas as pd
import transform
from sklearn.ensemble import RandomForestClassifier 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, jaccard_similarity_score, accuracy_score
import scipy as sp

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
dfPredicted.to_csv('data/results2.csv', sep=',', index=False)
