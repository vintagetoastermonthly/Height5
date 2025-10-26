#!/bin/bash
# Run all tests with coverage

echo "Running unit tests..."
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

echo ""
echo "Coverage report generated in htmlcov/index.html"
