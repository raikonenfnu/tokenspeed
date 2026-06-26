# General Agent Guidelines

> If a `AGENTS.local.md` file exists alongside this file, read and respect it--
> it contains developer-specific overrides that supplement this shared guidance.

## Development environment

* Before any work, check local Python venv and activate if one exists.
* Don't install pip packages outside the local Python venv if one exists.

## Code changes

* Add tests and update docs for the changed code.
* Before creating commits, run `pre-commit run --all-files` to format.
* When creating commits, perform sign off on behalf of the author.

## Dependency boundaries

* `tokenspeed` runtime dependencies should stay vendor-neutral.
* Runtime code should use `tokenspeed-kernel` as its only kernel package
  boundary.
* Third-party kernel libraries belong under `tokenspeed-kernel`; avoid direct
  runtime dependencies or imports that bypass it.
* If a dependency repeatedly breaks during version upgrades or slows project
  progress, consider removing it entirely or at least making it optional.

## tokenspeed-kernel

Inside the root tokenspeed-kernel/ directory:

* All direct tokenspeed-triton imports should happen in `_triton.py` and then
  re-import to other places.
* All direct third-party code should be placed in `thirdparty/` and imported
  into `ops/` then registered via `register_kernel`.
* Prefer CuteDSL for NVIDIA GPU kernels and Triton Gluon for AMD GPU kernels.
  Use Triton for portable solutions across vendors. Vendor libraries should
  stay optional, and other solutions may be used as temporary transitions, but
  new work should consolidate toward these backend choices.
* Files under `ops/` should follow `<family>/<solution>` structure, like
  `gemm/trtllm.py` or `attention/triton/`.
* When defining new public APIs, explain arguments and returns in docstring.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
