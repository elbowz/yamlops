CREATE KEYSPACE IF NOT EXISTS yamlops
  WITH replication = {'class':'SimpleStrategy', 'replication_factor':1};

CREATE TABLE IF NOT EXISTS yamlops.bracelet (
  id uuid,
  age float,
  sex float,
  cp float,
  trestbps float,
  chol float,
  fbs float,
  restecg float,
  thalach float,
  exang float,
  oldpeak float,
  slope float,
  ca float,
  thal float,
  num int,
  PRIMARY KEY ((id))
) WITH comment = 'Bracelet DataSet';