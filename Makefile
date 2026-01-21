# Gradience Development Makefile
.PHONY: verify-version install test lint clean help

help: ## Show this help message
	@echo "Gradience Development Commands:"
	@echo "================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

verify-version: ## Verify version consistency across all sources
	@echo "🔍 Verifying version consistency..."
	@python3 scripts/verify_version.py

install: ## Install package in development mode
	@echo "📦 Installing Gradience in development mode..."
	@pip3 install -e .

test: ## Run tests
	@echo "🧪 Running tests..."
	@python3 -m pytest tests/ -v

lint: ## Run linting checks
	@echo "🔍 Running linting checks..."
	@python3 -m ruff check gradience/
	@python3 -m mypy gradience/

clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf build/ dist/ *.egg-info/
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -delete

# Development workflow
dev-install: install verify-version ## Install and verify for development

# Release workflow  
pre-release: verify-version test lint ## Run all checks before release
	@echo "✅ All pre-release checks passed!"

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