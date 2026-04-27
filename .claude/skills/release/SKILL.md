---
name: release
description: Use when releasing a new version of notectl — bumps version in Cargo.toml, updates Cargo.lock, commits, pushes, tags, and pushes the tag.
---

# release

Release a new version of notectl. Accepts one argument: `major`, `minor`, or `patch`.

## Steps

1. **Parse the current version** from `[workspace.package] version` in `Cargo.toml`.
   ```bash
   grep '^version' Cargo.toml   # e.g. version = "0.9.0"
   ```

2. **Compute the next version** from the argument:
   - `major` → increment first component, reset others to 0 (0.9.0 → 1.0.0)
   - `minor` → increment second component, reset patch to 0 (0.9.0 → 0.10.0)
   - `patch` → increment third component (0.9.0 → 0.9.1)

3. **Update `Cargo.toml`** — change the single `version = "X.Y.Z"` line under `[workspace.package]`. All workspace members inherit it via `version.workspace = true`.

4. **Regenerate `Cargo.lock`** and verify the build:
   ```bash
   cargo update --workspace
   cargo build --workspace
   ```

5. **Commit the version bump**:
   ```bash
   git add Cargo.toml Cargo.lock
   git commit -m "Bump to vX.Y.Z"
   ```

6. **Push** the commit to origin:
   ```bash
   git push
   ```

7. **Create and push the tag**:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

## Preconditions

- Working tree must be clean before starting (the version bump commit should contain only `Cargo.toml` and `Cargo.lock`). If there are uncommitted changes, commit or stash them first.
- `cargo build --workspace` must succeed before tagging.

## Example

```
/release minor   # 0.9.0 → 0.10.0, tag v0.10.0
/release patch   # 0.9.0 → 0.9.1,  tag v0.9.1
/release major   # 0.9.0 → 1.0.0,  tag v1.0.0
```
