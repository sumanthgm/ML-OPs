import joblib

from scipy.stats import randint

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier

if __name__ == '__main__':
    dataset = fetch_openml(data_id=42890, as_frame=True, parser='auto')
    maintenance_data = dataset.data
    target = 'Machine failure'
    numeric_features = [
        'Air temperature [K]', 
        'Process temperature [K]', 
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    categorical_features = ['Type']

    X = maintenance_data.drop(
        columns=[target, 'UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    )
    y = maintenance_data[target]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    Xtrain.to_csv('data/20230921_training_features.csv', index=False)
    ytrain.to_csv('data/20230921_training_target.csv', index=False)

    Xtest.to_csv('data/20230921_test_features.csv', index=False)
    ytest.to_csv('data/20230921_test_target.csv', index=False)

    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown='ignore'), categorical_features)
    )

    model_gbr = GradientBoostingClassifier(
        max_depth=3, 
        n_estimators=100, 
        random_state=42
    )

    model_pipeline = make_pipeline(
        preprocessor, 
        model_gbr
    )
    
    param_distrib = {
        "gradientboostingclassifier__max_depth": randint(3, 12),
        "gradientboostingclassifier__n_estimators": randint(100, 1000)
    }

    rand_search_cv = RandomizedSearchCV(
        model_pipeline,
        param_distrib,
        n_iter=2,
        cv=3,
        random_state=42
    )

    rand_search_cv.fit(Xtrain, ytrain)

    with open('metrics/metrics-model-v1.txt', 'w') as metrics_file:
        metrics_file.write(f"Best validation accuracy = {rand_search_cv.best_score_}")
    
    print(f"Best validation accuracy = {rand_search_cv.best_score_}")
    
    saved_model_path = "models/model-v1.joblib"
    joblib.dump(rand_search_cv.best_estimator_, saved_model_path)