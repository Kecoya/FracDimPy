# Contributing to FracDimPy

Thank you for your interest in contributing to FracDimPy! This document provides guidelines for contributors.

## Development Setup

1. **Fork the repository** and clone your fork locally
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   pip install -e .[dev]
   ```

4. **Run tests** to ensure everything is working:
   ```bash
   pytest
   ```

## Running Tests

Run all tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=src/fracdimpy --cov-report=term-missing
```

Run specific test files:
```bash
pytest tests/test_monofractal/test_box_counting.py
```

Run performance benchmarks:
```bash
pytest tests/benchmarks/ --benchmark-only
```

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking

Format your code:
```bash
black src/ tests/
```

Run linting:
```bash
flake8 src/
```

Run type checking:
```bash
mypy src/
```

## Adding New Features

1. **Add tests** for your new functionality
2. **Update documentation** with clear examples
3. **Ensure all tests pass** before submitting a pull request
4. **Follow the existing code style**

## Testing Requirements

- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test components working together
- **Performance tests**: Ensure new features don't significantly impact performance

Aim for >80% test coverage.

## Documentation

- Add docstrings to all public functions following the NumPy style
- Include examples in docstrings
- Update README.md if adding major new features

## Submitting Pull Requests

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit them with descriptive messages

3. **Push to your fork** and create a pull request

4. **Fill out the pull request template** with:
   - Description of changes
   - Testing performed
   - Any breaking changes

## Issue Reporting

When reporting bugs, please include:
- Python version
- FracDimPy version
- Operating system
- Minimal reproducible example
- Expected vs. actual behavior

## License

By contributing to FracDimPy, you agree that your contributions will be licensed under the GPL-3.0 license.