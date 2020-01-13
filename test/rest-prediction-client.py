import click
import requests

testFeatures = [
    [58,1,3,112,230,0,2,165,0,2.5,2,1,7],
    [35,0,4,138,183,0,0,182,0,1.4,1,0,3],
    [63,1,4,130,330,1,2,132,1,1.8,1,3,7],
    [65,1,4,135,254,0,2,127,0,2.8,2,1,7],
    [48,1,4,130,256,1,2,150,1,0,1,2,7],
    [63,0,4,150,407,0,2,154,0,4,2,3,7],
    [51,1,3,100,222,0,0,143,1,1.2,2,0,3],
    [55,1,4,140,217,0,0,111,1,5.6,3,0,7],
    [65,1,1,138,282,1,2,174,0,1.4,2,1,3],
    [85,1,0,128,222,1,0,134,0,1.2,1,0,2]
]

"""
eg.
 python rest-prediction-client.py --url 'http://localhost:5000/predict/bracelet'
 python rest-prediction-client.py --url 'http://localhost:5000/predict/bracelet' --manual '{"age": 5.109, "sex": 6.701, "cp": 8.666, "trestbps": 1.805, "chol": -1.376, "fbs": 2.353, "restecg": 1.033, "thalach": -2.665, "exang": -7.027, "oldpeak": -6.41, "slope": 5.946, "ca": 1.358, "thal": 5.7121}'
"""

@click.command(help="Make REST call to end point")
@click.option("--manual", default=False)
@click.option("--url", default='http://localhost:5000/predict/bracelet')
def call_rest(manual, url):

    if not manual:
        for cols in testFeatures:
            jsonRequest = { "age": cols[0], "sex": cols[1], "cp": cols[2],
                            "trestbps": cols[3], "chol": cols[4], "fbs": cols[5],
                            "restecg": cols[6], "thalach": cols[7], "exang": cols[8],
                            "oldpeak": cols[9], "slope": cols[10], "ca": cols[11], "thal": cols[12] }

            print("Send JSON request: %s" % jsonRequest)

            res = requests.post(url, json=jsonRequest)

            if res.ok:
                print("REST service answer: %s\n" % res.json())
    else:
        import json

        print("Sent these features: %s" % json.loads(manual))

        res = requests.post(url, json=json.loads(manual))

        if res.ok:
            print("REST service answer: %s" % res.json())


if __name__ == "__main__":
    call_rest()