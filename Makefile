.PHONY: help install install-dev setup prepare-data train evaluate test test-cov lint format run-app docker-build docker-run docker-build-train docker-push clean clean-data clean-models clean-all

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
DOCKER := docker
PROJECT_NAME := vehicle-detection
DOCKER_IMAGE := $(PROJECT_NAME):latest
DOCKER_IMAGE_TRAIN := $(PROJECT_NAME):train

# Colors for output
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_BLUE := \033[34m

##@ Installation

install: ## Install production dependencies
	@echo "$(COLOR_BLUE)📦 Installing production dependencies...$(COLOR_RESET)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(COLOR_GREEN)✅ Installation complete!$(COLOR_RESET)"

install-dev: ## Install development dependencies
	@echo "$(COLOR_BLUE)📦 Installing development dependencies...$(COLOR_RESET)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "$(COLOR_GREEN)✅ Development installation complete!$(COLOR_RESET)"

##@ Project Setup

setup: ## Create project directory structure
	@echo "$(COLOR_BLUE)📁 Creating project directories...$(COLOR_RESET)"
	mkdir -p data/raw/train/images data/raw/train/labels
	mkdir -p data/processed/train/images data/processed/train/labels
	mkdir -p data/processed/val/images data/processed/val/labels
	mkdir -p data/processed/test/images data/processed/test/labels
	mkdir -p models logs
	touch data/raw/.gitkeep
	touch data/processed/.gitkeep
	touch models/.gitkeep
	touch logs/.gitkeep
	@echo "$(COLOR_GREEN)✅ Project structure created!$(COLOR_RESET)"

##@ Data Management

prepare-data: ## Prepare dataset for training
	@echo "$(COLOR_BLUE)🔄 Preparing dataset...$(COLOR_RESET)"
	$(PYTHON) -m src.data_preparation
	@echo "$(COLOR_GREEN)✅ Data preparation complete!$(COLOR_RESET)"

##@ Training & Evaluation

train: ## Train the model
	@echo "$(COLOR_BLUE)🚀 Starting model training...$(COLOR_RESET)"
	$(PYTHON) -m src.train
	@echo "$(COLOR_GREEN)✅ Training complete!$(COLOR_RESET)"

evaluate: ## Evaluate the trained model
	@echo "$(COLOR_BLUE)🔍 Evaluating model...$(COLOR_RESET)"
	$(PYTHON) -m src.evaluate
	@echo "$(COLOR_GREEN)✅ Evaluation complete!$(COLOR_RESET)"

##@ Testing

test: ## Run tests
	@echo "$(COLOR_BLUE)🧪 Running tests...$(COLOR_RESET)"
	pytest tests/ -v --tb=short
	@echo "$(COLOR_GREEN)✅ Tests complete!$(COLOR_RESET)"

test-cov: ## Run tests with coverage report
	@echo "$(COLOR_BLUE)🧪 Running tests with coverage...$(COLOR_RESET)"
	pytest tests/ --cov=src --cov=app --cov-report=html --cov-report=term --cov-report=xml
	@echo "$(COLOR_GREEN)✅ Coverage report generated!$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)📊 Open htmlcov/index.html to view detailed report$(COLOR_RESET)"

test-quick: ## Run tests quickly (no coverage)
	@echo "$(COLOR_BLUE)⚡ Running quick tests...$(COLOR_RESET)"
	pytest tests/ -v -x --tb=short
	@echo "$(COLOR_GREEN)✅ Quick tests complete!$(COLOR_RESET)"

##@ Code Quality

lint: ## Run code linting
	@echo "$(COLOR_BLUE)🔍 Running linters...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)→ Running flake8...$(COLOR_RESET)"
	flake8 src/ app/ tests/ --max-line-length=100 --ignore=E203,W503 --exclude=__pycache__ || true
	@echo "$(COLOR_GREEN)✅ Linting complete!$(COLOR_RESET)"

format: ## Format code with black and isort
	@echo "$(COLOR_BLUE)✨ Formatting code...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)→ Running black...$(COLOR_RESET)"
	black src/ app/ tests/
	@echo "$(COLOR_YELLOW)→ Running isort...$(COLOR_RESET)"
	isort src/ app/ tests/
	@echo "$(COLOR_GREEN)✅ Code formatted!$(COLOR_RESET)"

format-check: ## Check code formatting without modifying
	@echo "$(COLOR_BLUE)🔍 Checking code format...$(COLOR_RESET)"
	black --check --diff src/ app/ tests/
	isort --check-only --diff src/ app/ tests/

##@ Application

run-app: ## Run the Gradio application
	@echo "$(COLOR_BLUE)🚀 Starting Gradio application...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)📱 App will be available at: http://localhost:7860$(COLOR_RESET)"
	$(PYTHON) app/app.py

run-app-public: ## Run app with public URL (using Gradio share)
	@echo "$(COLOR_BLUE)🚀 Starting Gradio application with public URL...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)🌐 Generating public URL...$(COLOR_RESET)"
	$(PYTHON) -c "import app.app as a; a.demo.launch(share=True)"

##@ Docker

docker-build: ## Build production Docker image
	@echo "$(COLOR_BLUE)🐳 Building production Docker image...$(COLOR_RESET)"
	$(DOCKER) build -f docker/Dockerfile -t $(DOCKER_IMAGE) .
	@echo "$(COLOR_GREEN)✅ Docker image built: $(DOCKER_IMAGE)$(COLOR_RESET)"

docker-run: ## Run Docker container
	@echo "$(COLOR_BLUE)🐳 Running Docker container...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)📱 App will be available at: http://localhost:7860$(COLOR_RESET)"
	$(DOCKER) run -p 7860:7860 --name $(PROJECT_NAME)-app $(DOCKER_IMAGE)

docker-run-detached: ## Run Docker container in background
	@echo "$(COLOR_BLUE)🐳 Running Docker container in background...$(COLOR_RESET)"
	$(DOCKER) run -d -p 7860:7860 --name $(PROJECT_NAME)-app $(DOCKER_IMAGE)
	@echo "$(COLOR_GREEN)✅ Container running in background$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)📱 Access at: http://localhost:7860$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)📋 Logs: make docker-logs$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)🛑 Stop: make docker-stop$(COLOR_RESET)"

docker-stop: ## Stop running Docker container
	@echo "$(COLOR_BLUE)🛑 Stopping Docker container...$(COLOR_RESET)"
	$(DOCKER) stop $(PROJECT_NAME)-app || true
	$(DOCKER) rm $(PROJECT_NAME)-app || true
	@echo "$(COLOR_GREEN)✅ Container stopped$(COLOR_RESET)"

docker-logs: ## View Docker container logs
	@echo "$(COLOR_BLUE)📋 Viewing container logs...$(COLOR_RESET)"
	$(DOCKER) logs -f $(PROJECT_NAME)-app

docker-shell: ## Open shell in running container
	@echo "$(COLOR_BLUE)🐚 Opening shell in container...$(COLOR_RESET)"
	$(DOCKER) exec -it $(PROJECT_NAME)-app /bin/bash

docker-build-train: ## Build training Docker image
	@echo "$(COLOR_BLUE)🐳 Building training Docker image...$(COLOR_RESET)"
	$(DOCKER) build -f docker/Dockerfile.train -t $(DOCKER_IMAGE_TRAIN) .
	@echo "$(COLOR_GREEN)✅ Training Docker image built: $(DOCKER_IMAGE_TRAIN)$(COLOR_RESET)"

docker-push: ## Push Docker image to registry (requires login)
	@echo "$(COLOR_BLUE)🚀 Pushing Docker image...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)⚠️  Make sure you're logged in: docker login$(COLOR_RESET)"
	$(DOCKER) tag $(DOCKER_IMAGE) yourusername/$(DOCKER_IMAGE)
	$(DOCKER) push yourusername/$(DOCKER_IMAGE)
	@echo "$(COLOR_GREEN)✅ Image pushed!$(COLOR_RESET)"

docker-clean: ## Remove Docker images
	@echo "$(COLOR_BLUE)🧹 Cleaning Docker images...$(COLOR_RESET)"
	$(DOCKER) rmi $(DOCKER_IMAGE) || true
	$(DOCKER) rmi $(DOCKER_IMAGE_TRAIN) || true
	@echo "$(COLOR_GREEN)✅ Docker images removed$(COLOR_RESET)"

##@ Cleanup

clean: ## Clean cache and temporary files
	@echo "$(COLOR_BLUE)🧹 Cleaning cache files...$(COLOR_RESET)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .tox
	rm -rf dist
	rm -rf build
	rm -f logs/*.log
	@echo "$(COLOR_GREEN)✅ Cache cleaned!$(COLOR_RESET)"

clean-data: ## Clean processed data (keeps raw data)
	@echo "$(COLOR_BLUE)🧹 Cleaning processed data...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)⚠️  This will delete data/processed/* (raw data preserved)$(COLOR_RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/processed/*; \
		touch data/processed/.gitkeep; \
		echo "$(COLOR_GREEN)✅ Processed data cleaned!$(COLOR_RESET)"; \
	else \
		echo "$(COLOR_YELLOW)❌ Cancelled$(COLOR_RESET)"; \
	fi

clean-models: ## Clean trained models
	@echo "$(COLOR_BLUE)🧹 Cleaning models...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)⚠️  This will delete models/*.pt$(COLOR_RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf models/*.pt; \
		rm -rf models/*.onnx; \
		touch models/.gitkeep; \
		echo "$(COLOR_GREEN)✅ Models cleaned!$(COLOR_RESET)"; \
	else \
		echo "$(COLOR_YELLOW)❌ Cancelled$(COLOR_RESET)"; \
	fi

clean-all: clean clean-data clean-models docker-clean ## Clean everything (cache, data, models, Docker)
	@echo "$(COLOR_GREEN)✅ Everything cleaned!$(COLOR_RESET)"

##@ Git Operations

git-status: ## Show git status
	@echo "$(COLOR_BLUE)📊 Git Status:$(COLOR_RESET)"
	@git status

git-add-all: ## Stage all changes
	@echo "$(COLOR_BLUE)➕ Staging all changes...$(COLOR_RESET)"
	git add .
	@echo "$(COLOR_GREEN)✅ Changes staged!$(COLOR_RESET)"

git-commit: ## Commit with message (use MSG="your message")
	@echo "$(COLOR_BLUE)💾 Committing changes...$(COLOR_RESET)"
	@if [ -z "$(MSG)" ]; then \
		echo "$(COLOR_YELLOW)⚠️  Usage: make git-commit MSG=\"your commit message\"$(COLOR_RESET)"; \
		exit 1; \
	fi
	git commit -m "$(MSG)"
	@echo "$(COLOR_GREEN)✅ Committed!$(COLOR_RESET)"

git-push: ## Push to remote
	@echo "$(COLOR_BLUE)🚀 Pushing to remote...$(COLOR_RESET)"
	git push
	@echo "$(COLOR_GREEN)✅ Pushed to remote!$(COLOR_RESET)"

deploy: git-add-all ## Quick deploy: add all, commit, and push (use MSG="your message")
	@echo "$(COLOR_BLUE)🚀 Deploying...$(COLOR_RESET)"
	@if [ -z "$(MSG)" ]; then \
		echo "$(COLOR_YELLOW)⚠️  Usage: make deploy MSG=\"your commit message\"$(COLOR_RESET)"; \
		exit 1; \
	fi
	@make git-commit MSG="$(MSG)"
	@make git-push
	@echo "$(COLOR_GREEN)✅ Deployment started! Check GitHub Actions$(COLOR_RESET)"

##@ Information

check-gpu: ## Check if GPU is available
	@echo "$(COLOR_BLUE)🖥️  Checking GPU availability...$(COLOR_RESET)"
	@$(PYTHON) -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

check-env: ## Check environment setup
	@echo "$(COLOR_BLUE)🔍 Environment Check:$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Python Version:$(COLOR_RESET)"
	@$(PYTHON) --version
	@echo "$(COLOR_YELLOW)Pip Version:$(COLOR_RESET)"
	@$(PIP) --version
	@echo "$(COLOR_YELLOW)Docker Version:$(COLOR_RESET)"
	@$(DOCKER) --version || echo "Docker not installed"
	@echo "$(COLOR_YELLOW)CUDA Available:$(COLOR_RESET)"
	@$(PYTHON) -c "import torch; print(torch.cuda.is_available())" || echo "PyTorch not installed"

tree: ## Show project structure
	@echo "$(COLOR_BLUE)📁 Project Structure:$(COLOR_RESET)"
	@tree -L 3 -I '__pycache__|*.pyc|.git|venv|env' . || ls -R

size: ## Show project size
	@echo "$(COLOR_BLUE)📊 Project Size:$(COLOR_RESET)"
	@du -sh . 2>/dev/null || echo "du command not available"
	@echo "$(COLOR_YELLOW)Breakdown:$(COLOR_RESET)"
	@du -sh data models logs 2>/dev/null || echo "Directories not found"

##@ Help

help: ## Display this help message
	@echo "$(COLOR_BOLD)$(COLOR_BLUE)Vehicle Detection MLOps - Makefile Commands$(COLOR_RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(COLOR_YELLOW)<target>$(COLOR_RESET)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(COLOR_GREEN)%-20s$(COLOR_RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(COLOR_BOLD)%s$(COLOR_RESET)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(COLOR_BOLD)Examples:$(COLOR_RESET)"
	@echo "  make install          # Install dependencies"
	@echo "  make setup            # Create project structure"
	@echo "  make train            # Train the model"
	@echo "  make test             # Run tests"
	@echo "  make run-app          # Start Gradio app"
	@echo "  make docker-build     # Build Docker image"
	@echo "  make deploy MSG=\"Update model\"  # Commit and deploy"
	@echo ""
	@echo "$(COLOR_BOLD)Quick Start:$(COLOR_RESET)"
	@echo "  1. make install       # Install dependencies"
	@echo "  2. make setup         # Setup directories"
	@echo "  3. Upload dataset to data/raw/train/"
	@echo "  4. make prepare-data  # Process dataset"
	@echo "  5. make train         # Train model"
	@echo "  6. make run-app       # Test locally"
	@echo "  7. make deploy MSG=\"Initial deployment\""
	@echo ""