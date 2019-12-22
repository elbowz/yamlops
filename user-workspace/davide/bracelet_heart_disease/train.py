import click

import pandas as pd
import numpy as np

# tracking and versioning by mlflow
import mlflow
import mlflow.sklearn

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

@click.command(help="Do training on ./cleveland.csv, version and track params and metrics")
@click.option("--data-set", default='./cleveland.csv')
@click.option("--kernel", default='rbf')
@click.option("--degree", default=3)
def train(data_set, kernel, degree):

    # Data loading

    df = pd.read_csv(data_set, header = None)

    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
                  'fbs', 'restecg', 'thalach', 'exang',
                  'oldpeak', 'slope', 'ca', 'thal', 'target']

    ### 1 = male, 0 = female
    df.isnull().sum()

    df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
    df['sex'] = df.sex.map({0: 'female', 1: 'male'})
    df['thal'] = df.thal.fillna(df.thal.mean())
    df['ca'] = df.ca.fillna(df.ca.mean())
    df['sex'] = df.sex.map({'female': 0, 'male': 1})

    # Data preprocessing

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    from sklearn.preprocessing import StandardScaler as ss
    sc = ss()
    #X_train = sc.fit_transform(X_train)
    #X_test = sc.transform(X_test)

    np.savetxt(data_set + '_processed', X_train, delimiter=",")

    # Training

    with mlflow.start_run():

        from sklearn.svm import SVC
        classifier = SVC(kernel = kernel, degree = degree)
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        from sklearn.metrics import confusion_matrix
        cm_test = confusion_matrix(y_pred, y_test)

        y_pred_train = classifier.predict(X_train)
        cm_train = confusion_matrix(y_pred_train, y_train)

        accuracy_test_set = (cm_test[0][0] + cm_test[1][1])/len(y_test)
        accuracy_training_set = (cm_train[0][0] + cm_train[1][1])/len(y_train)
        (rmse, mae, r2) = eval_metrics(y_test, y_pred)

        # Log parameter, metrics, and model to MLflow
        mlflow.log_param("kernel", kernel)
        mlflow.log_param("degree", degree)

        mlflow.log_metric("Accuracy", accuracy_test_set)
        mlflow.log_metric("Accuracy TrainingSet", accuracy_training_set)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Add tag to MLflow log
        mlflow.set_tag('model', 'SVM')
        mlflow.set_tag('stage', 'prod')

        # Add DataSet to artifacts
        mlflow.log_artifact('./cleveland.csv')

        mlflow.sklearn.log_model(classifier, "model")

        # Print out metrics
        print("SVM model (kernel=%s, degree=%s):" % (kernel, degree))
        print('  Accuracy for TestSet %s' % accuracy_test_set)
        print('  Accuracy for TrainingSet %s' % accuracy_training_set)
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

if __name__ == "__main__":
    train()