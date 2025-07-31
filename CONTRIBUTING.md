# Contributing to Grid-Fed-RL-Gym

Thank you for your interest in contributing to Grid-Fed-RL-Gym! This document provides guidelines for contributing to this federated reinforcement learning framework for power distribution networks.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic understanding of reinforcement learning and power systems (helpful but not required)

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/grid-fed-rl-gym.git
   cd grid-fed-rl-gym
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Run tests to verify setup**
   ```bash
   pytest
   ```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following our [coding standards](#coding-standards)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run tests
   pytest
   
   # Check code quality
   pre-commit run --all-files
   
   # Check type hints
   mypy grid_fed_rl/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Pull Request Process

1. **Before submitting**
   - [ ] Tests pass locally
   - [ ] Code follows style guidelines
   - [ ] Documentation is updated
   - [ ] Commit messages are descriptive

2. **Pull request requirements**
   - Use the provided PR template
   - Link related issues
   - Provide clear description of changes
   - Include screenshots for UI changes

3. **Review process**
   - Maintainers will review within 48 hours
   - Address feedback promptly
   - Squash commits before merge

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Import organization**: Use `isort` via Ruff
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all public methods

### Code Quality Tools

- **Ruff**: Linting and formatting
- **MyPy**: Static type checking
- **Bandit**: Security analysis
- **Pre-commit**: Automated quality checks

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `GridEnvironment`)
- **Functions/Variables**: `snake_case` (e.g., `get_action`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_VOLTAGE`)
- **Private methods**: `_leading_underscore`

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for combined functionality
└── fixtures/       # Test data and fixtures
```

### Writing Tests

1. **Test naming**: `test_<functionality>_<expected_behavior>`
2. **Test organization**: One test class per module
3. **Fixtures**: Use pytest fixtures for reusable test data
4. **Mocking**: Mock external dependencies appropriately

### Test Coverage

- Minimum 80% coverage required
- Focus on critical paths and edge cases
- Test both success and failure scenarios

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/unit/test_algorithms.py

# With coverage
pytest --cov=grid_fed_rl

# Integration tests only
pytest -m integration
```

## Documentation

### Documentation Types

1. **API Documentation**: Automatically generated from docstrings
2. **User Guides**: Step-by-step tutorials and examples
3. **Architecture Documentation**: High-level system design
4. **Developer Documentation**: Contributing and development guides

### Writing Documentation

- **Docstrings**: Google style with examples
- **README updates**: Keep installation and usage current
- **Examples**: Include working code examples
- **Architecture**: Update diagrams for structural changes

### Building Documentation

```bash
cd docs/
make html
```

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Pull Requests**: Code contributions and technical discussion

### Getting Help

1. **Check existing issues** before creating new ones
2. **Use issue templates** for bug reports and feature requests
3. **Be specific** about your environment and use case
4. **Provide examples** when possible

### Contribution Ideas

Looking for ways to contribute? Consider:

- **Bug fixes**: Check open issues labeled `bug`
- **Features**: Implement requested features
- **Documentation**: Improve examples and guides
- **Testing**: Add test coverage for untested areas
- **Performance**: Optimize algorithms and simulations

## Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Changelog

All notable changes are documented in [CHANGELOG.md](CHANGELOG.md) following [Keep a Changelog](https://keepachangelog.com/) format.

## License

By contributing to Grid-Fed-RL-Gym, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to reach out via:
- GitHub Issues for technical questions
- GitHub Discussions for general discussion
- Email maintainers for private inquiries

Thank you for contributing to making power grids smarter and more resilient! 🔌⚡