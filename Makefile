.PHONY: help build run stop clean logs test

help:
	@echo "Telecom Churn Prediction System - Make Commands"
	@echo "================================================"
	@echo "make build          - Build Docker image"
	@echo "make run            - Start all services"
	@echo "make stop           - Stop all services"
	@echo "make clean          - Remove containers and volumes"
	@echo "make logs           - Show logs"
	@echo "make test           - Run tests"
	@echo "make scale          - Scale workers"
	@echo "make generate-data  - Generate sample data"
	@echo "make train          - Train model"
	@echo "make predict        - Run predictions"
	@echo "make full-pipeline  - Run complete pipeline"

build:
	docker build -t telecom-churn-backend:latest .

run:
	docker-compose up -d
	@echo "Services started!"
	@echo "Spark Master UI: http://localhost:8080"
	@echo "API Server: http://localhost:5001"

stop:
	docker-compose down

clean:
	docker-compose down -v
	rm -rf data/ logs/

logs:
	docker-compose logs -f

test:
	pytest tests/ -v --cov=main

scale:
	docker-compose up -d --scale spark-worker=4

generate-data:
	docker exec spark-master python /opt/app/main.py generate-data --records 100000

train:
	docker exec spark-master python /opt/app/main.py train

predict:
	docker exec spark-master python /opt/app/main.py predict

full-pipeline:
	docker exec spark-master python /opt/app/main.py full-pipeline --records 50000
