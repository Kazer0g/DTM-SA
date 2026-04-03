# Developer Guide for DTM-SA Repository

## Overview

This guide outlines the workflow, branching strategy, and best practices for contributing to the Distributed Text Mining and Sentiment Analysis (DTM-SA) project. Follow these guidelines to ensure smooth collaboration and maintain code quality.

## Branching Strategy

### Main Branches
- **`main`**: The production-ready branch. Only stable, tested versions are merged here. Each working version is pushed from the `dev` branch after thorough review.
- **`dev`**: The development branch. This is where ongoing development happens. Changes are pushed here from developer-specific branches.

### Developer Branches
- **Naming Convention**: Branches should be named after the developer's nickname (e.g., `Kazer0g`, `illmmmiira`).
- **Purpose**: These are personal branches for individual developers. Push your changes here first to publish them.
- **Merging**: Only changes from these named branches can be pushed to `dev`. You can merge any necessary branches into your personal branch, but it's recommended to create separate branches for different tasks (e.g., `Kazer0g/New-branch`) and merge them into your personal branch before pushing to `dev`.

### Feature/Task Branches
- **Naming Convention**: Use the format `nickname-feature-name` (e.g., `Kazer0g-New-branch`).
- Create short-lived branches for specific tasks from your personal branch.
- Merge these into your personal branch when ready, then push your personal branch to `dev`.
- Do not push feature branches directly to the remote repository. Keep them local or merge them into your personal branch.

### Publishing to Remote
- Only push personal branches (named after developers) to the remote.
- These branches will eventually be merged into `dev`, and from `dev` into `main`.
- Avoid pushing temporary or feature branches to the remote to keep the repository clean.

## Workflow

1. **Start a Task**:
   - Create a feature branch from your personal branch: `git checkout -b Kazer0g-New-branch`.

2. **Develop and Commit**:
   - Make changes and commit with proper messages (see Commit Message Policy below).
   - Merge your feature branch back into your personal branch: `git checkout yournickname && git merge yournickname-task-name`.

3. **Publish Changes**:
   - Push your personal branch to remote: `git push origin yournickname`.
   - Create a pull request (PR) from your personal branch to `dev` for review.

4. **Merge to Dev**:
   - After PR approval, merge into `dev`.
   - From `dev`, create a PR to `main` when ready for release.

5. **Clean Up**:
   - Delete local feature branches after merging: `git branch -d Kazer0g-New-branch`.
   - Prune remote branches if needed: `git remote prune origin`.

## Commit Message Policy

Follow the Conventional Commits format for clear, structured commit messages. Start each commit with a type from the table below, followed by a colon and a brief description.

### Commit Types

| Type     | Description |
|----------|-------------|
| `build`  | Changes affecting the build system or external dependencies |
| `chore`  | Routine tasks that don't change code, docs, or features |
| `ci`     | Changes to CI/CD configuration files and scripts |
| `docs`   | Documentation changes only |
| `feat`   | New features in the product |
| `fix`    | Bug fixes |
| `perf`   | Performance improvements |
| `refactor` | Code refactoring (no bug fixes or new features) |
| `revert` | Reverting a previous commit |
| `style`  | Code style changes (whitespace, formatting, etc.) |
| `test`   | Adding or fixing tests |

### Examples
- `feat: add sentiment analysis pipeline`
- `fix: resolve null pointer in word frequency counter`
- `docs: update README with setup instructions`
- `refactor: simplify MapReduce job structure`

Keep messages concise but descriptive. Use the imperative mood (e.g., "add" instead of "added"). If needed, add a body with more details after a blank line.

## Additional Guidelines
- Always pull the latest changes before starting work: `git pull origin dev`.
- Use `git rebase` for clean history when merging feature branches.
- Run tests and linters before pushing.
- Communicate with the team via PR comments for reviews.

For questions, refer to the README.md or contact the team lead.
