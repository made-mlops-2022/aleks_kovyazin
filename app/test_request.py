from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {'health check': 'OK'}

def test_predict():
    response = client.post(
        "/predict",
        headers={"X-Token": "coneofsilence"},
        json={"text": "foobar"}
    )
    assert response.status_code == 200
    assert float(response.json()['result']) > 0.5

def test_train():
    response = client.post(
        "/train",
        headers={"X-Token": "coneofsilence"},
        json={"text": "foobar"}
    )
    assert response.status_code == 200
    assert response.json() == {'result': 'Model is done !'}


if __name__=='__main__':
    test_health()
    test_train()
    test_predict()
