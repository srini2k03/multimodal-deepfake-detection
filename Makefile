# Common developer tasks. Run `make help` for a list.
PY ?= python
BACKEND := backend
CONFIG ?= configs/default.yaml

.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format test test-fast api frontend-install frontend-dev frontend-build \
        extract extract-audio train-xception train-inception train-temporal train-audio train-all evaluate \
        predict smoke docker-build docker-up docker-down clean fetch-weights serve report bench-asvspoof bench-dff \
        frontend-smoke

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install full backend runtime (TensorFlow included)
	cd $(BACKEND) && $(PY) -m pip install -r requirements.txt -r requirements-dev.txt

install-dev: ## Install lightweight dev deps (no TensorFlow; TF tests are skipped)
	cd $(BACKEND) && $(PY) -m pip install -r requirements-dev.txt

lint: ## Ruff lint + format check
	cd $(BACKEND) && ruff check . && ruff format --check .

format: ## Auto-format backend code
	cd $(BACKEND) && ruff format . && ruff check --fix .

test: ## Run all backend tests
	cd $(BACKEND) && $(PY) -m pytest

test-fast: ## Run tests that do not need TensorFlow
	cd $(BACKEND) && $(PY) -m pytest -m "not tf"

api: ## Run the Flask dev server on :5000
	cd $(BACKEND) && $(PY) -m deepfake_detector.api --config $(CONFIG) --debug

frontend-install: ## Install frontend dependencies
	cd frontend && npm install

frontend-dev: ## Run the Vite dev server on :5173 (proxies /api to :5000)
	cd frontend && npm run dev

frontend-build: ## Production build of the frontend
	cd frontend && npm run build

extract: ## Extract aligned face crops for the configured dataset
	cd $(BACKEND) && $(PY) -m deepfake_detector.preprocessing.extract faces --config $(CONFIG) --workers 4

extract-audio: ## Compute log-mel features for the audio dataset
	cd $(BACKEND) && $(PY) -m deepfake_detector.preprocessing.extract audio --config $(CONFIG)

train-xception: ## Two-stage fine-tuning of the Xception frame model
	cd $(BACKEND) && $(PY) -m deepfake_detector.training.train --config $(CONFIG) --model xception

train-inception: ## Two-stage fine-tuning of the InceptionV3 frame model
	cd $(BACKEND) && $(PY) -m deepfake_detector.training.train --config $(CONFIG) --model inception_v3

train-temporal: ## Attention-pooling temporal head on top of the Xception backbone
	cd $(BACKEND) && $(PY) -m deepfake_detector.training.train --config $(CONFIG) --model temporal --backbone xception

train-audio: ## Audio CNN branch
	cd $(BACKEND) && $(PY) -m deepfake_detector.training.train --config $(CONFIG) --model audio

train-all: train-xception train-inception train-temporal ## Train all visual models

evaluate: ## Evaluate on the test split (writes reports/)
	cd $(BACKEND) && $(PY) -m deepfake_detector.evaluation.evaluate --config $(CONFIG) --split test --output reports

predict: ## Predict a single file: make predict FILE=path/to/video.mp4
	cd $(BACKEND) && $(PY) -m deepfake_detector.inference.cli $(abspath $(FILE)) --config $(CONFIG)

smoke: ## End-to-end CPU smoke run on synthetic data (pipeline check only, not a real model)
	cd $(BACKEND) && $(PY) scripts/make_synthetic_dataset.py --out data/synthetic --videos 12 \
	  && $(PY) -m deepfake_detector.preprocessing.extract faces --config configs/smoke.yaml --workers 2 --keep-faceless \
	  && $(PY) -m deepfake_detector.preprocessing.extract audio --config configs/smoke.yaml \
	  && $(PY) -m deepfake_detector.training.train --config configs/smoke.yaml --model xception \
	  && $(PY) -m deepfake_detector.training.train --config configs/smoke.yaml --model inception_v3 \
	  && $(PY) -m deepfake_detector.training.train --config configs/smoke.yaml --model temporal \
	  && $(PY) -m deepfake_detector.training.train --config configs/smoke.yaml --model audio \
	  && $(PY) -m deepfake_detector.evaluation.evaluate --config configs/smoke.yaml --output data/smoke/reports

fetch-weights: ## Download the released benchmark models (checksum-verified) + model card into backend/weights
	cd $(BACKEND) && $(PY) scripts/fetch_weights.py

serve: ## Run the API with the released models (run fetch-weights first)
	cd $(BACKEND) && DFD_CONFIG=configs/serve.yaml $(PY) -m deepfake_detector.api

report: ## Rebuild docs/BENCHMARKS.md tables, docs/figures and the site data from results/
	cd $(BACKEND) && $(PY) scripts/make_report.py

LA_ZIP ?= data/LA.zip
WORK ?= data/work
bench-asvspoof: ## ASVspoof 2019 LA benchmark locally: make bench-asvspoof LA_ZIP=/path/to/LA.zip
	cd $(BACKEND) && for step in "features --archive $(LA_ZIP) --work $(WORK)" "train --work $(WORK) --system cnn" \
	  "train --work $(WORK) --system cnn_trim" "evaluate --archive $(LA_ZIP) --work $(WORK) --out ../results/audio/asvspoof2019_la"; do \
	  $(PY) -m deepfake_detector.benchmarks.run_asvspoof $$step || exit 1; done

DFF_DIR ?= data/dff
bench-dff: ## DeepFakeFace benchmark locally: make bench-dff DFF_DIR=/dir/with/the/four/zips
	cd $(BACKEND) && for step in "prepare --zips $(DFF_DIR) --work $(WORK)/dff" "probe --work $(WORK)/dff" \
	  "finetune --work $(WORK)/dff" "evaluate --work $(WORK)/dff --out ../results/visual/dff"; do \
	  $(PY) -m deepfake_detector.benchmarks.run_dff $$step || exit 1; done

frontend-smoke: ## Build the Pages demo and run the Playwright smoke test
	cd frontend && npm run build:pages && node scripts/smoke.mjs

docker-build: ## Build API and web images
	docker compose build

docker-up: ## Start the stack (web on :8080, API on :5000)
	docker compose up -d

docker-down: ## Stop the stack
	docker compose down

clean: ## Remove caches and build output
	rm -rf $(BACKEND)/.pytest_cache $(BACKEND)/.ruff_cache frontend/dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
