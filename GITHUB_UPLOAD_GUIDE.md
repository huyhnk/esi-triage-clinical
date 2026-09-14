# Quick GitHub upload guide

## Option 1: GitHub website

1. Create a new empty repository on GitHub.
2. Extract this ZIP file.
3. Upload the repository files and folders.
4. Do not upload anything inside `data/raw/`, `data/processed/`, `models/`, or generated `outputs/` folders.
5. Commit the files.

## Option 2: Git command line

```bash
git init
git add .
git commit -m "Initial commit: ESI triage prediction pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Before `git add .`, verify that no restricted clinical data are present in the repository.
