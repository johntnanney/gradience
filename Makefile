# Gradience Development Makefile
.PHONY: setup setup-cache verify-version install test lint format clean help demo-gain-audit sensitivity-check build publish publish-test docs docs-serve

help: ## Show this help message
	@echo "Gradience Development Commands:"
	@echo "================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

setup: ## Complete development setup (creates venv, installs [hf,dev])
	@echo "🚀 Setting up Gradience development environment..."
	@python3 -m venv .venv
	@. .venv/bin/activate && pip install -U pip && pip install -e ".[hf,dev]"
	@echo "✅ Setup complete! Activate with: source .venv/bin/activate"
	@echo ""
	@echo "💡 Consider running 'make setup-cache' to configure storage"

setup-cache: ## Configure cache directories to prevent disk space issues
	@echo "🗂️  Configuring cache environment..."
	@bash scripts/setup_cache_env.sh

verify-version: ## Verify version consistency across all sources
	@echo "🔍 Verifying version consistency..."
	@python3 scripts/verify_version.py

install: ## Install package in development mode ([hf,dev] extras)
	@echo "📦 Installing Gradience with [hf,dev] extras..."
	@pip3 install -e ".[hf,dev]"

test: ## Run tests with coverage
	@echo "🧪 Running tests..."
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && python -m pytest tests/ -v --cov=gradience --cov-report=term-missing; \
	else \
		python3 -m pytest tests/ -v --cov=gradience --cov-report=term-missing; \
	fi

test-quick: ## Run tests without coverage (faster)
	@echo "🧪 Running tests (quick)..."
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && python -m pytest tests/ -q; \
	else \
		python3 -m pytest tests/ -q; \
	fi

test-smoke: ## Run CPU-only smoke tests (no GPU required, ~6 seconds)
	@echo "💨 Running CPU smoke tests..."
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && python scripts/run_ci_smoke_tests.py --timing; \
	else \
		python3 scripts/run_ci_smoke_tests.py --timing; \
	fi

lint: ## Run linting checks only
	@echo "🔍 Running linting checks..."
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && ruff check . && mypy gradience/; \
	else \
		python3 -m ruff check . && python3 -m mypy gradience/; \
	fi

format: ## Format code with ruff
	@echo "🎨 Formatting code..."
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && ruff format .; \
	else \
		python3 -m ruff format .; \
	fi

format-check: ## Check code formatting without making changes
	@echo "🎨 Checking code format..."
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && ruff format --check .; \
	else \
		python3 -m ruff format --check .; \
	fi

clean: ## Clean build artifacts and caches
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf build/ dist/ *.egg-info/
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -delete
	@find . -name ".pytest_cache" -exec rm -rf {} +
	@find . -name ".coverage" -delete
	@find . -name ".mypy_cache" -exec rm -rf {} +

# Development workflows
dev-install: install verify-version ## Install and verify for development
	@echo "✅ Development installation complete!"

demo-gain-audit: ## Demo the LoRA gain audit functionality (v0.7.0)
	@echo "🎯 Running LoRA gain audit demo..."
	@./scripts/demo_gain_audit.sh

sensitivity-check: ## Prove gain metrics respond to rank changes (r=4 vs r=16)  
	@echo "🔬 Running sensitivity analysis..."
	@./scripts/sensitivity_check_fixed_seed.sh

docs: ## Build documentation site
	@echo "📚 Building documentation..."
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && mkdocs build; \
	else \
		python3 -m mkdocs build; \
	fi
	@echo "✅ Documentation built in site/"

docs-serve: ## Serve documentation locally (http://127.0.0.1:8000)
	@echo "📚 Serving documentation at http://127.0.0.1:8000..."
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && mkdocs serve; \
	else \
		python3 -m mkdocs serve; \
	fi

check: lint format-check test-quick ## Run all code quality checks
	@echo "✅ All checks passed!"

# Release workflow  
pre-release: verify-version test lint ## Run all checks before release
	@echo "✅ All pre-release checks passed!"

build: clean ## Build sdist and wheel
	@echo "📦 Building package..."
	@python3 -m build
	@echo "✅ Build complete. Artifacts in dist/"

publish: pre-release build ## Build and publish to PyPI
	@echo "🚀 Publishing to PyPI..."
	@python3 -m twine check dist/*
	@python3 -m twine upload dist/*
	@echo "✅ Published to PyPI!"

publish-test: pre-release build ## Build and publish to TestPyPI
	@echo "🧪 Publishing to TestPyPI..."
	@python3 -m twine check dist/*
	@python3 -m twine upload --repository testpypi dist/*
	@echo "✅ Published to TestPyPI!"
	@echo "Install with: pip install -i https://test.pypi.org/simple/ gradience==0.11.0"

# Version management
bump-patch: ## Bump patch version (0.4.2 -> 0.4.3)
	@echo "🔢 This would bump patch version - manual implementation needed"
	@echo "   1. Update pyproject.toml version"
	@echo "   2. Run: make dev-install"
	@echo "   3. Run: make verify-version"
	@echo "   4. Commit and tag"

bump-minor: ## Bump minor version (0.4.2 -> 0.5.0)
	@echo "🔢 This would bump minor version - manual implementation needed"
	@echo "   1. Update pyproject.toml version"
	@echo "   2. Run: make dev-install"
	@echo "   3. Run: make verify-version"
	@echo "   4. Commit and tag"