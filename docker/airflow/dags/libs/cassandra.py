from cassandra.cluster import Cluster
from cassandra.policies import RoundRobinPolicy

def cassandra_connect(host='cassandra', keyspace='yamlops'):
    cluster = Cluster([host], load_balancing_policy=RoundRobinPolicy())
    return cluster.connect(keyspace)

def cassandra_insert_bracelet(session, value):
    session.execute(
        """
        INSERT INTO bracelet (id, age, sex, cp, trestbps, chol,
                                fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        VALUES (uuid(), %(age)s, %(sex)s, %(cp)s, %(trestbps)s,
                %(chol)s, %(fbs)s, %(restecg)s, %(thalach)s, %(exang)s,
                %(oldpeak)s, %(slope)s, %(ca)s, %(thal)s)
        """,
        value
    )