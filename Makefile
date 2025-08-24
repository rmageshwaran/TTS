# TTS Virtual Microphone - Development Makefile
# 
# This Makefile provides common development tasks for the project.
# Run 'make help' to see available commands.

.PHONY: help install install-dev clean test test-unit test-integration test-e2e
.PHONY: lint format type-check security-check pre-commit setup-dev
.PHONY: coverage coverage-html docs docs-serve build package
.PHONY: run run-api run-gui docker-build docker-run

# Configuration
PYTHON := python3
PIP := pip3
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy
BANDIT := bandit

# Directories
SRC_DIRS := core config utils plugins interfaces virtual_audio
TEST_DIRS := tests
DOCS_DIR := docs

# Default target
help: ## Show this help message
	@echo "TTS Virtual Microphone - Development Commands"
	@echo "============================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# Installation
install: ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	pre-commit install

setup-dev: install-dev ## Complete development environment setup
	@echo "Setting up development environment..."
	@mkdir -p logs models cache temp tests/logs
	@echo "Development environment ready!"

# Cleaning
clean: ## Clean temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf logs/*.log
	rm -rf temp/*

# Testing
test: ## Run all tests
	$(PYTEST) $(TEST_DIRS) -v

test-unit: ## Run unit tests only
	$(PYTEST) $(TEST_DIRS)/unit -v -m "not slow"

test-integration: ## Run integration tests
	$(PYTEST) $(TEST_DIRS)/integration -v

test-e2e: ## Run end-to-end tests
	$(PYTEST) $(TEST_DIRS)/e2e -v -m "slow"

test-fast: ## Run fast tests only (exclude slow tests)
	$(PYTEST) $(TEST_DIRS) -v -m "not slow"

test-slow: ## Run slow tests only
	$(PYTEST) $(TEST_DIRS) -v -m "slow"

test-audio: ## Run audio-related tests (requires audio hardware)
	$(PYTEST) $(TEST_DIRS) -v -m "audio"

test-gpu: ## Run GPU tests (requires CUDA)
	$(PYTEST) $(TEST_DIRS) -v -m "gpu"

# Code Quality
lint: ## Run all linting checks
	$(FLAKE8) $(SRC_DIRS) $(TEST_DIRS)
	$(BLACK) --check $(SRC_DIRS) $(TEST_DIRS)
	$(ISORT) --check-only $(SRC_DIRS) $(TEST_DIRS)

format: ## Format code with black and isort
	$(BLACK) $(SRC_DIRS) $(TEST_DIRS) main.py
	$(ISORT) $(SRC_DIRS) $(TEST_DIRS) main.py

type-check: ## Run type checking with mypy
	$(MYPY) $(SRC_DIRS) main.py

security-check: ## Run security checks with bandit
	$(BANDIT) -r $(SRC_DIRS) -f json -o security-report.json || true
	$(BANDIT) -r $(SRC_DIRS)

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

quality-check: lint type-check security-check ## Run all quality checks

# Coverage
coverage: ## Run tests with coverage
	$(PYTEST) $(TEST_DIRS) --cov=$(SRC_DIRS) --cov-report=term-missing --cov-report=xml

coverage-html: ## Generate HTML coverage report
	$(PYTEST) $(TEST_DIRS) --cov=$(SRC_DIRS) --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

# Documentation
docs: ## Build documentation
	cd $(DOCS_DIR) && make html

docs-serve: ## Serve documentation locally
	cd $(DOCS_DIR) && make livehtml

docs-clean: ## Clean documentation build files
	cd $(DOCS_DIR) && make clean

# Application Running
run: ## Run the main application (CLI)
	$(PYTHON) main.py

run-interactive: ## Run in interactive mode
	$(PYTHON) main.py --interactive

run-api: ## Run API server
	$(PYTHON) main.py --api --port 8080

run-gui: ## Run GUI interface
	$(PYTHON) main.py --gui

run-test: ## Run system tests
	$(PYTHON) main.py --test

list-devices: ## List available audio devices
	$(PYTHON) main.py --list-devices

# Development utilities
profile: ## Run performance profiling
	$(PYTHON) -m cProfile -o profile.stats main.py --test
	$(PYTHON) -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

benchmark: ## Run benchmarks
	$(PYTEST) tests/performance/ -v --benchmark-only

memory-profile: ## Run memory profiling
	$(PYTHON) -m memory_profiler main.py --test

# Building and Packaging
build: clean ## Build distribution packages
	$(PYTHON) -m build

package: build ## Create distribution package
	@echo "Package created in dist/"

install-package: package ## Install the built package
	$(PIP) install dist/*.whl

# Docker (if using containerization)
docker-build: ## Build Docker image
	docker build -t tts-virtual-mic .

docker-run: ## Run Docker container
	docker run -it --rm --device /dev/snd tts-virtual-mic

# Git hooks
install-hooks: ## Install git hooks
	pre-commit install
	pre-commit install --hook-type pre-push

# Environment management
create-env: ## Create virtual environment
	$(PYTHON) -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

freeze: ## Freeze current dependencies
	$(PIP) freeze > requirements-freeze.txt

# Development workflow
dev-setup: create-env install-dev setup-dev ## Complete development setup from scratch
	@echo "Development environment fully set up!"

dev-test: format lint type-check test-fast ## Quick development test cycle

dev-check: format lint type-check coverage ## Full development check

ci-test: lint type-check security-check coverage ## CI/CD test pipeline

release-check: clean ci-test build ## Pre-release checks

# Troubleshooting
debug: ## Run application in debug mode
	$(PYTHON) main.py --debug --verbose --test

check-audio: ## Check audio system
	$(PYTHON) main.py --list-devices --test-audio

check-dependencies: ## Check if all dependencies are installed
	$(PIP) check
	$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"
	$(PYTHON) -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	$(PYTHON) -c "import sounddevice; print(f'SoundDevice: {sounddevice.__version__}')"

system-info: ## Show system information
	@echo "Python version:"
	$(PYTHON) --version
	@echo "Pip version:"
	$(PIP) --version
	@echo "Git version:"
	git --version
	@echo "Available audio devices:"
	$(PYTHON) main.py --list-devices

# Performance testing
perf-test: ## Run performance tests
	$(PYTEST) tests/performance/ -v

load-test: ## Run load tests
	$(PYTEST) tests/performance/test_load.py -v

# Utility targets
watch-tests: ## Watch files and run tests on changes
	$(PYTEST) $(TEST_DIRS) -f

serve-coverage: coverage-html ## Serve coverage report
	$(PYTHON) -m http.server 8000 -d htmlcov

init-project: ## Initialize new project (run once)
	@echo "Initializing TTS Virtual Microphone project..."
	git init
	mkdir -p logs models cache temp tests/logs
	touch logs/.gitkeep models/.gitkeep cache/.gitkeep temp/.gitkeep
	@echo "Project initialized!"

# Help for specific topics
help-testing: ## Show testing help
	@echo "Testing Commands:"
	@echo "  test          - Run all tests"
	@echo "  test-unit     - Run unit tests only" 
	@echo "  test-fast     - Run fast tests (exclude slow ones)"
	@echo "  coverage      - Run tests with coverage report"
	@echo "  benchmark     - Run performance benchmarks"

help-quality: ## Show code quality help
	@echo "Code Quality Commands:"
	@echo "  format        - Format code with black and isort"
	@echo "  lint          - Run linting checks"
	@echo "  type-check    - Run type checking"
	@echo "  security-check- Run security analysis"
	@echo "  pre-commit    - Run all pre-commit hooks"

help-dev: ## Show development help
	@echo "Development Commands:"
	@echo "  setup-dev     - Set up development environment"
	@echo "  dev-test      - Quick development test cycle"
	@echo "  dev-check     - Full development validation"
	@echo "  run           - Run the application"
	@echo "  debug         - Run in debug mode"

# Version information
version: ## Show version information
	@echo "TTS Virtual Microphone v1.0.0"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Build date: $(shell date)"

.DEFAULT_GOAL := help