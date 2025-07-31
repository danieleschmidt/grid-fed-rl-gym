# Development Guide

This guide covers setting up a development environment and contributing to grid-fed-rl-gym.

## Prerequisites

- Python 3.9+ (3.11 recommended)
- pip or conda for package management
- Git for version control

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/terragonlabs/grid-fed-rl-gym.git
cd grid-fed-rl-gym

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

## Development Workflow

### Code Quality

We use several tools to maintain code quality:

- **ruff**: Linting and code formatting
- **mypy**: Static type checking
- **black**: Code formatting (integrated with ruff)
- **pytest**: Testing framework
- **pre-commit**: Git hooks for quality checks

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=grid_fed_rl

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run specific test file
pytest tests/unit/test_version.py
```

### Code Formatting and Linting

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Fix auto-fixable issues
ruff check . --fix

# Type checking
mypy grid_fed_rl/
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[dev]"

# Build documentation (when implemented)
cd docs/
make html
```

## Project Structure

```
grid-fed-rl-gym/
├── grid_fed_rl/           # Main package
│   ├── algorithms/        # RL algorithms
│   ├── environments/      # Grid environments
│   ├── federated/         # Federated learning
│   ├── feeders/           # Grid network definitions
│   ├── controllers/       # Grid controllers
│   ├── evaluation/        # Metrics and evaluation
│   └── utils/             # Utility functions
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test data
├── docs/                  # Documentation
└── scripts/               # Development scripts
```

## Testing Guidelines

- Write tests for all new features
- Maintain >80% test coverage
- Use descriptive test names
- Group related tests in classes
- Use fixtures for common test data
- Mark slow tests with `@pytest.mark.slow`

## Code Style Guidelines

- Follow PEP 8 style guide
- Use type hints for all functions
- Write docstrings for public APIs
- Keep functions focused and small
- Use descriptive variable names
- Add comments for complex logic

## Submitting Changes

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes with tests
3. Run the full test suite: `pytest`
4. Ensure code quality: `ruff check . && mypy grid_fed_rl/`
5. Commit with descriptive messages
6. Push and create a pull request

## Performance Considerations

- Profile code before optimizing
- Use vectorized operations with NumPy
- Consider GPU acceleration for training
- Monitor memory usage for large datasets
- Cache expensive computations

## Debugging Tips

- Use `pytest -s` to see print statements
- Set breakpoints with `import pdb; pdb.set_trace()`
- Use logging instead of print for debugging
- Test individual components in isolation
- Check input/output shapes for tensor operations

## Getting Help

- Check existing issues on GitHub
- Read the documentation
- Look at test files for usage examples
- Join discussions in GitHub Discussions
- Contact maintainers for security issues