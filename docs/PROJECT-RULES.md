# Project Rules and Workflow

## Overview

This document describes how to use the project board, issue lifecycle, and milestone planning for the DTM-SA (Distributed Text Mining and Sentiment Analysis) project.

## Issue Board Workflow

We use a Kanban-style board with four columns to track task progress. Each task (issue) moves through these stages based on its status.

### Board Columns

1. **Backlog**
   - New issues start here.
   - These are tasks that have been identified but are not yet scheduled for active work.
   - Tasks remain here until they are added to a milestone.

2. **Todo**
   - Issues assigned to the current milestone are moved here.
   - These are tasks ready to be worked on in the current sprint/week.
   - Team members pick tasks from this column to start development.

3. **In Progress**
   - A task moves here when a pull request (PR) is created for it.
   - This indicates active work is underway and a code review is pending.
   - The PR must be linked to the issue for tracking.

4. **Done**
   - A task moves here when its PR is approved and merged.
   - This marks the task as complete and ready for production or the next milestone.

### Task Flow Details

- **Backlog → Todo**: When a task is added to a milestone during sprint planning, move it to "Todo".
- **Todo → In Progress**: When you create a PR for a task, move it to "In Progress".
- **In Progress → Done**: When the PR is approved and merged, move it to "Done".
- **In Progress → Todo (Declined PR)**: If a PR is declined or needs major changes, move the issue back to "Todo" for re-work.

## Milestones

The project is organized into four one-week milestones, all scheduled in April 2026.

| Milestone | Dates | Description |
|-----------|-------|-------------|
| **Week 0 (0.4)** | April 0-7, 2026 | Project initialization, setup, documentation, and foundational work |
| **Week 1 (1.4)** | April 8-14, 2026 | Core feature development and initial implementation |
| **Week 2 (2.4)** | April 15-21, 2026 | Feature refinement, testing, and optimization |
| **Week 3 (3.4)** | April 22-28, 2026 | Final polish, integration testing, and deployment preparation |

### Milestone Assignment

- When planning a week's work, assign relevant issues to the corresponding milestone.
- Move assigned issues from "Backlog" to "Todo".
- Team members should focus on completing issues in the current milestone first.

## Using the Board

### For Team Members

1. Check the board regularly (preferably daily) to stay updated on project status.
2. Pick a task from the "Todo" column.
3. Create a branch from your personal developer branch following the naming convention: `YourNickname-task-description`.
4. Work on the task, commit your changes with proper commit messages (see GUIDE.md).
5. When done, create a PR linking to the issue and move it to "In Progress".
6. After PR approval, the task moves to "Done".

### For Reviewers/Leads

1. Monitor the "In Progress" column for PRs awaiting review.
2. Review code, request changes if needed, or approve.
3. Approved PRs are merged, and issues are moved to "Done".
4. If changes are requested, the issue is moved back to "Todo".

## Best Practices

- **Keep the board updated**: Ensure the board reflects the current state of work. Move issues immediately when their status changes.
- **Link PRs to issues**: Always reference the issue number in your PR (e.g., "Fixes #15" or "Resolves #42").
- **One task per PR**: Try to address one issue per pull request for clarity and easier review.
- **Communicate delays**: If a task is blocked or delayed, update the issue description or add a comment.
- **Use labels**: Tag issues with labels (e.g., "bug", "documentation", "refactor") for better organization.

## Additional Resources

- **GUIDE.md**: Developer workflow and branching strategy
- **README.md**: Project overview and validation checklist
- For questions about specific workflows, refer to the issue labels and milestone descriptions.
