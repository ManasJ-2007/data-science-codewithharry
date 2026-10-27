# Git, Colab and tools

## Git workflow
```
working directory ──git add──▶ staging area ──git commit──▶ local repo ──git push──▶ GitHub
```
| Task | Command |
|---|---|
| Start / copy a repo | `git init`, `git clone <url>` |
| See what changed | `git status`, `git diff`, `git diff --staged`, `git log --oneline` |
| Save work | `git add <file>` or `git add .`, then `git commit -m "Explain why"` |
| Undo | `git restore <file>` (discard edits), `git restore --staged <file>` (unstage), `git revert <commit>` (safe undo of a pushed commit) |
| Branches | `git switch -c feature`, then `git merge feature`; a pull request on GitHub for review |
| Remote | `git remote add origin <url>`, `git push -u origin main`, `git pull` |
| Ignore files | `.gitignore` for `__pycache__/`, `.venv/`, `.env`, large data and model files |

Good habits: small commits with clear messages, never commit secrets, and don't rewrite history that others have pulled.

## Google Colab
A free hosted Jupyter with GPUs. Mount Drive for data (`drive.mount("/content/drive")`), use `!pip install` for extra packages, and switch the runtime type to GPU for deep learning and Whisper.

## Jupyter and conda
Notebooks are for exploration. Move finished code into `.py` modules so it can be imported, tested and run end to end, as every project in this repo is.
