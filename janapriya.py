import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = {'length_of_email':[100,500,200,1000,300,50,800,600,100,900],
        'number_of_links':[2,5,1,8,3,0,7,4,1,6],
        'contains_spammy_words':[1,0,1,1,0,1,1,0,1,1],
        'is_fraud':[1,0,1,1,0,0,1,0,0,1]
        }
df = pd.DataFrame(data)
X = df[['length_of_email','number_of_links','contains_spammy_words']]
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=0)
print(f'Accuracy:{accuracy*100:.2f}%')
print('Confusion Matrix:\n', conf_matrix)
print('classification_report:\n', class_report)
