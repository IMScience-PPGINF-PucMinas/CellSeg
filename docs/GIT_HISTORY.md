# Shrinking Git history after cleanup

The working tree was reduced to Oral Epithelium + core scripts, but **old commits may still reference multi-GB SIBGRAPI outputs** inside `.git`.

Before the first push of the reorganized repo:

1. Ensure large folders are deleted locally (already done by `tools/reorganize_oral_repo.sh`).
2. Commit the new layout.
3. If `du -sh .git` is still huge (>500 MB), use [git-filter-repo](https://github.com/newren/git-filter-repo) or BFG to strip blobs, **or** create a fresh orphan branch:

```bash
git checkout --orphan oral-epithelium-main
git add -A
git commit -m "Reorganize: Oral Epithelium DB + pipeline only"
# force-push only after team agreement
```

Never force-push `main` without coordinating with collaborators.
