# Git workflow: one branch and one PR per ticket

`main` is protected: it changes only through a PR whose CI matrix (Windows, macOS, Ubuntu) is green. The repo is public, so everything pushed (branches, commits, PR titles and bodies) is public too.

## Steps

1. **Branch.** From an up-to-date `main`, create `<NN>-<slug>`, named after the ticket file (`.scratch/<feature>/issues/06-paths-adapter.md` → `06-paths-adapter`). Work that has no ticket gets a short descriptive name (`docs-git-workflow`). Done when `git branch --show-current` names the ticket branch.
2. **Commit.** Commit on the ticket branch. The pre-commit hooks must pass on every commit, never bypassed, so each commit stands on its own. Done when the ticket's work is committed and `git status` is clean.
3. **Push.** Run `git push -u origin <branch>`, naming the branch. The pre-push hook runs the test suite first. Done when the branch is on `origin`.
4. **Open the PR** against `main`. Use the `pr` skill for the body. Done when the PR exists and CI is running.
5. **Merge.** Once all three OS jobs are green, use **Rebase and merge**, then delete the branch. Rebase keeps each commit (rule 14 relies on separate import and cuts commits for review). Done when the commits are on `main` and the branch is gone.
6. **Close the ticket.** Set its `Status:` line in `.scratch/` and record any follow-ups there. Done when the ticket file reflects the merge.

A red CI run means the ticket is still open. Fix it with new commits on the same branch.

## Public text

PR titles, bodies and branch names follow the same privacy rules as commit messages (CLAUDE.md → Privacy). Tickets stay in `.scratch/`, so refer to one by number and title only ("Ticket 06: paths adapter"). For a ticket on the fatigue path, the PR states the replay gate result as "replay gate: byte-identical (run locally)", with no corpus details.

## Branches that stay local

`pre-scrub-backup` holds the history from before the scrub and its private data. It is local only. Push one named branch at a time; `git push --all` and `git push --mirror` would publish it.

## Ordering

- Branches for tickets that touch disjoint files may be open at once, each in its own worktree. The Phase 3 adapters (06–10) are like this.
- Tickets that share files go in sequence: branch the next one from `main` after the previous one merges. Head pose (11) touches `face_mesh`, `cv_pipeline` and `calibration`; 12 → 13 → 14 run in order.
- A release is a `v*` tag on `main` after the release ticket's PR merges. The tag triggers `release.yml`.
