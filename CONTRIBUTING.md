# Contributing to Agentic EDA Pipeline

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites
- Python 3.10+
- Git
- [Ollama](https://ollama.com) (for LLM features)

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/Sujay1709/agentic-eda-pipeline.git
cd agentic-eda-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies in development mode
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock pylint flake8 black

# Copy environment template
cp .env.example .env
```

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

Follow these guidelines:

**Code Style:**
- Use [Black](https://github.com/psf/black) for formatting
- Follow PEP 8 conventions
- Add type hints where possible
- Write descriptive docstrings

**Example:**
```python
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

def process_data(
    input_path: str,
    output_dir: str,
    verbose: bool = False
) -> Dict[str, str]:
    """
    Process input data and generate outputs.
    
    Args:
        input_path: Path to input CSV file
        output_dir: Directory to save outputs
        verbose: Enable verbose logging
    
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Processing file: {input_path}")
    # Implementation here
    return {"status": "complete"}
```

### 3. Write/Update Tests

All new features must include tests:

```bash
# Create test file in tests/
# tests/test_new_feature.py

import pytest
from new_module import new_function

class TestNewFeature:
    def test_basic_functionality(self):
        result = new_function()
        assert result is not None
    
    def test_edge_case(self):
        with pytest.raises(ValueError):
            new_function(invalid_input)
```

Run tests:
```bash
pytest tests/ -v --cov=.
```

### 4. Format and Lint

```bash
# Format code with Black
black .

# Check for issues with Pylint
pylint **/*.py

# Check for style issues
flake8 .
```

### 5. Update Documentation

- Update `README.md` if adding user-facing features
- Update docstrings in code
- Add examples if applicable
- Update `CONTRIBUTING.md` if process changes

### 6. Commit Changes

```bash
git add .
git commit -m "feat: add new visualization agent

- Implement new agent for advanced plots
- Add comprehensive tests
- Update documentation"
```

**Commit Message Format:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Tests
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `chore:` - Maintenance

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Create a Pull Request with:
- Clear description of changes
- Reference to related issues
- Screenshots/results if applicable
- Checklist of testing performed

## Pull Request Process

1. **Automated Checks**: PR must pass GitHub Actions workflows
2. **Code Review**: At least one maintainer approval required
3. **Tests**: All tests must pass
4. **Coverage**: Maintain or improve code coverage
5. **Documentation**: All changes documented

## Areas for Contribution

### High Priority
- [ ] Add more visualization types
- [ ] Implement caching for repeated analyses
- [ ] Add support for more data formats (Parquet, HDF5, SQL databases)
- [ ] Create interactive dashboards for results
- [ ] Add anomaly detection features

### Medium Priority
- [ ] Performance optimizations
- [ ] Additional LLM model support
- [ ] Advanced missing data imputation
- [ ] Statistical test suite
- [ ] Data quality scoring

### Documentation
- [ ] Architecture diagrams
- [ ] API documentation
- [ ] Tutorial notebooks
- [ ] Video walkthroughs

## Reporting Bugs

Use the issue template and provide:
- Description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error logs or stack traces

## Questions or Discussions

- Open a GitHub Discussion for questions
- Check existing issues first
- Join the community chat (if available)

## Code of Conduct

- Be respectful and inclusive
- Welcome diverse perspectives
- Help others learn
- Report issues constructively

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for making this project better! 🎉
