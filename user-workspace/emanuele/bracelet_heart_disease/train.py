"""
mlflow.run('/tmp/work/sklearn_elasticnet_wine', 'main', parameters={"alpha": 0.66}, use_conda=False)
mlflow run ./bracelet_heart_disease -P kernel=rbf -P degree=5 --no-conda --experiment-name 'bracelet'

mlflow models serve -m  /mlflow/artifacts/2/c0844c7cb78040b38dcc30a932d73a9f/artifacts/model -p 1234 --no-conda
mlflow models serve -m  runs:/c0844c7cb78040b38dcc30a932d73a9f/model -p 1234 --no-conda

curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],"data":[[58,1,3,112,230,0,2,165,0,2.5,2,1,7],[58,1,3,112,230,0,2,165,0,2.5,2,1,7],[35,0,4,138,183,0,0,182,0,1.4,1,0,3],[63,1,4,130,330,1,2,132,1,1.8,1,3,7],[65,1,4,135,254,0,2,127,0,2.8,2,1,7],[48,1,4,130,256,1,2,150,1,0,1,2,7],[63,0,4,150,407,0,2,154,0,4,2,3,7],[51,1,3,100,222,0,0,143,1,1.2,2,0,3],[55,1,4,140,217,0,0,111,1,5.6,3,0,7],[65,1,1,138,282,1,2,174,0,1.4,2,1,3]]}' http://127.0.0.1:1234/invocations

curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],"data":[[-1.131852084261172342e+00,6.701505774131562010e-01,8.666591890729338221e-01,-1.278857280950254305e+00,-9.569777236582576174e-01,-4.248385633132285566e-01,1.033692316610911766e+00,1.243173822960859765e+00,-7.027283689263056354e-01,-8.981430174089009011e-01,-9.866752405934120507e-01,3.212770880583482591e-01,-9.788821281571468136e-01],[58,1,3,112,230,0,2,165,0,2.5,2,1,7],[35,0,4,138,183,0,0,182,0,1.4,1,0,3],[63,1,4,130,330,1,2,132,1,1.8,1,3,7],[65,1,4,135,254,0,2,127,0,2.8,2,1,7],[48,1,4,130,256,1,2,150,1,0,1,2,7],[63,0,4,150,407,0,2,154,0,4,2,3,7],[51,1,3,100,222,0,0,143,1,1.2,2,0,3],[55,1,4,140,217,0,0,111,1,5.6,3,0,7],[65,1,1,138,282,1,2,174,0,1.4,2,1,3],
[7.286212937466987616e-02,6.701505774131562010e-01,8.666591890729338221e-01,1.576683058503165480e+00,8.788099757668194068e-01,-4.248385633132285566e-01,1.033692316610911766e+00,-1.943561285112772596e-01,1.423024947075768987e+00,-2.309116737006769171e-01,5.946188536026522087e-01,3.212770880583482591e-01,1.087907688558129582e+00],
[-3.665734459222487474e-02,6.701505774131562010e-01,8.666591890729338221e-01,-7.077492130595703923e-01,-1.136565650775928082e+00,-4.248385633132285566e-01,-9.836749464523192321e-01,-1.631886079983414284e+00,-7.027283689263056354e-01,2.695118340804909529e-01,5.946188536026522087e-01,3.212770880583482591e-01,1.087907688558129582e+00],
[5.109400252422489075e-01,6.701505774131562010e-01,8.666591890729338221e-01,1.805126285659439000e+00,-1.376016220266155665e+00,2.353835283221942021e+00,1.033692316610911766e+00,-2.665110732604012878e+00,-7.027283689263056354e-01,-6.410383777362098356e-02,5.946188536026522087e-01,1.358162348438632261e+00,5.712102343793105108e-01]
]}' http://127.0.0.1:1234/invocations


"""

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