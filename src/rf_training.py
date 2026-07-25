import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
from joblib import dump

df = pd.read_csv('../data/data.csv')
df = df.drop('temp_C', axis=1)
df = df.rename(columns={col: col.replace('_seq', '_dg') for col in df.columns if col.endswith('_seq')})

class_col = 'class_2'

X = df.drop(columns=[class_col])
y = df[class_col]

random_state=100

X.replace([np.inf, -np.inf], np.nan, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

rf_model = RandomForestClassifier(
    n_estimators=250,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=random_state
)

# Metrics

k = 5  
kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='accuracy')

metrics_report = [f'K-Fold Cross-Validation Scores: {cv_scores}']
metrics_report.append(f'Mean CV Accuracy: {np.mean(cv_scores):.4f}')
metrics_report.append(f'Standard Deviation: {np.std(cv_scores):.4f}')

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

metrics_report.append('')
metrics_report.append(f'Final Test Set Accuracy: {accuracy_score(y_test, y_pred)}')
metrics_report.append('')
metrics_report.append('Classification Report:')
metrics_report.append(classification_report(y_test, y_pred))

permutation_result = permutation_importance(
    estimator=rf_model,
    X=X_test,
    y=y_test,
    scoring='accuracy',
    n_repeats=50,
    random_state=random_state,
    n_jobs=-1
)

permutation_importance = (
    pd.DataFrame({
        'feature': X_test.columns,
        'importance_mean': permutation_result.importances_mean,
        'importance_std': permutation_result.importances_std,
    })
    .sort_values('importance_mean', ascending=False)
    .reset_index(drop=True)
)

metrics_report.append('Feature Importances:')
metrics_report.append(str(permutation_importance))

dump(rf_model, '../models/rf_model.joblib')
with open('../models/rf_metrics.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(metrics_report))
