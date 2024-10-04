
# Contributing to Faster Whisper Server

Thank you for considering contributing to the Faster Whisper Server project! We welcome all contributions and feedback. Please follow this guide to help maintain a smooth contribution process.

## Table of Contents
- [How to Contribute](#how-to-contribute)
- [Setting Up Your Development Environment](#setting-up-your-development-environment)
- [Making Changes](#making-changes)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)
- [Code of Conduct](#code-of-conduct)
- [Getting Help](#getting-help)

## How to Contribute

1. Fork the repository to your own GitHub account.
2. Clone the forked repository to your local machine:
   ```bash
   git clone https://github.com/your-username/faster-whisper-server.git
   ```
3. Create a new branch for your contribution:
   ```bash
   git checkout -b feature/my-new-feature
   ```
4. Make your changes in the new branch.

## Setting Up Your Development Environment

Before contributing, ensure you have the project dependencies installed. You can quickly start the project with Docker.

1. **Install Docker**: Follow the [official guide](https://docs.docker.com/get-docker/) to install Docker if you haven’t already.
2. **Run with Docker**:
   - For **GPU support**:
     ```bash
     docker run --gpus=all --publish 8000:8000 --volume ~/.cache/huggingface:/root/.cache/huggingface fedirz/faster-whisper-server:latest-cuda
     ```
   - For **CPU support**:
     ```bash
     docker run --publish 8000:8000 --volume ~/.cache/huggingface:/root/.cache/huggingface fedirz/faster-whisper-server:latest-cpu
     ```
3. **Run with Docker Compose**:
   ```bash
   curl -sO https://raw.githubusercontent.com/fedirz/faster-whisper-server/master/compose.yaml
   docker compose up --detach faster-whisper-server-cuda
   # or
   docker compose up --detach faster-whisper-server-cpu
   ```

For live transcription, you can also explore the Kubernetes tutorial and other integrations listed in the README.

## Making Changes

1. Ensure your changes are well-tested.
2. Run formatting checks and tests locally before submitting:
   ```bash
   # Example commands
   pytest  # For running tests
   black .  # For formatting code using Black
   ```
3. Ensure that your code follows the style guidelines provided (e.g., PEP 8 for Python).

### Best Practices:
- **Write Clean Code**: Follow existing conventions and aim for readability and maintainability.
- **Document Your Code**: Add comments and documentation wherever necessary, especially for new features or complex sections.
- **Tests**: Add or update tests to cover your code changes. Make sure tests pass before submitting your changes.

## Submitting Pull Requests

Once your changes are ready:

1. Push your changes to your forked repository:
   ```bash
   git push origin feature/my-new-feature
   ```
2. Open a Pull Request (PR) from your forked repository to the main repository.
   - Add a clear title and description of your changes.
   - Reference any related issue numbers.
   - Ensure the PR description follows the [conventional commits format](https://www.conventionalcommits.org/en/v1.0.0/).

### Pull Request Guidelines:
- PRs should be focused on a single feature or fix.
- Ensure your PR title follows the commit message format (e.g., `fix: corrected a broken link` or `feat: added streaming support for large audio files`).
- Explain why your changes are necessary, and include any relevant context.
- Wait for a project maintainer to review your PR.

## Issue Reporting

If you encounter bugs or have general questions about the project, please open an issue on GitHub. Be sure to include:

1. **A descriptive title** summarizing the issue.
2. **Steps to reproduce** the bug or issue, including code snippets or logs.
3. **Expected behavior** and **actual behavior** for clarity.
4. **Environment details**, like operating system, Docker version, etc.

We welcome all feedback and reports to help us improve the project.

## Feature Requests

If you have ideas for new features, please submit them via a GitHub issue with the label `feature request`. When submitting a feature request:

- Explain **why** this feature would be helpful.
- Provide use cases or scenarios to clarify how the feature will benefit the project.
- Optionally, suggest how it could be implemented.

---

Thank you for contributing to Faster Whisper Server! Your support is greatly appreciated.
