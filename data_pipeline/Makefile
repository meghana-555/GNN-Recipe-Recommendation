.PHONY: help ingest-baseline run-generator run-batch

help:
	@echo "Mealie ML Data Team Workflow"
	@echo "----------------------------"
	@echo "Available commands:"
	@echo "  make ingest-baseline : Downloads from Kaggle, validates, uploads to S3"
	@echo "  make run-generator   : Simulates traffic and pushes Mealie JSON logs to S3"
	@echo "  make run-batch       : Merges data, runs evaluation logic, creates train/eval splits"

ingest-baseline:
	python ingest_baseline.py

run-generator:
	python generator.py

run-batch:
	python batch_pipeline.py
