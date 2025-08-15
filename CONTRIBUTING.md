# Contributing

Thanks for your interest in contributing! Below are guidelines to make contributions smooth and consistent.

How to contribute

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feat/my-feature
   ```
3. Make your changes. Keep commits small and focused.
4. Run tests locally (see `pytest`).
5. Push your branch to your fork and open a pull request (PR).

Commit messages

- Use clear, short commit messages (imperative mood).
- Provide a longer description in the commit body when needed.

Testing

- Add unit tests for any new functionality where possible.
- Run tests:
  ```bash
  pip install -r requirements.txt
  pytest -q
  ```

Code style

- Follow PEP8 formatting.
- Keep functions focused and small.
- Add docstrings for public functions/classes.

Security

- Do not include API keys, credentials, or sensitive files in commits/PRs.
- If you accidentally commit a secret, remove it and rotate the secret immediately.

Communication

- Include a clear description of what your PR changes and why.
- Link any related issues in the PR description.

Maintainers will review and provide feedback. Thanks!
