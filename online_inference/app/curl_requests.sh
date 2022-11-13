curl http://localhost:80/
curl -d '{"text":"value"}' -H "Content-Type: application/json" -X POST http://localhost:80/train
curl -d '{"text":"value"}' -H "Content-Type: application/json" -X POST http://localhost:80/predict

