# Versioning

`VERSION` is the canonical OrcaColony repository version. The same value is
shown in `README.md` and `PROGRESS_REPORT.md` and kept in `pyproject.toml`,
`src/orcacolony/__init__.py`, and the local project entry in `uv.lock`.

OrcaColony uses stable three-part semantic versions:

```text
MAJOR.MINOR.PATCH
```

Every project commit must raise the version. Ordinary incremental work raises
`PATCH`. A planned compatibility or project-phase change may raise `MINOR` or
`MAJOR` instead. Versions never move backward and are not reused.

GitHub pull requests and pushes to `main` run:

```bash
python scripts/check_version.py --base-ref <base-commit>
```

The check walks every new commit, verifies that all version locations
agree at that commit, and requires the version to be greater than its first
parent. A GitHub merge commit inherits the reviewed branch version and remains
greater than the previous `main` version.

Before committing local work, run:

```bash
python scripts/check_version.py --working-tree --base-ref HEAD
```

Release tags, when created, use `vMAJOR.MINOR.PATCH`.
