# Repository rules

These rules apply to anyone, human or tool, changing this repository. `AGENTS.md` is a link to this file.

## Releases

- A release is an annotated git tag `vMAJOR.MINOR.PATCH` on `main` (current release: `v0.1.0`).
- Every release has an entry in `CHANGELOG.md`, committed before the tag is created.
- Never force-push `main`. Never move or delete a published tag; fix mistakes with a new release.

## Authorship

- Commits and tags are authored as `Mladen Lotar <mladen@the-shop.hr>`.
- No AI attribution anywhere: commit messages, PR descriptions, docs, tag messages, code comments.
  That means no `Co-Authored-By` trailers for tools, no session or chat links, no tool or vendor
  signatures, and no narration of how the work was produced (agents, sessions, lanes, peers).
- Credit upstream work by name and link: the base model, heretic, llama.cpp and its PR authors,
  TensorFold (ashhart) and MLX (ml-explore). Commit messages end with
  "Builds on TensorFold by ashhart and llama.cpp by ggml-org." where relevant.

## Numbers

- Publish only numbers from runs validated by two independent harnesses, with a check that output
  tokens are identical across the compared configurations.
- State the machine, settings and whether the cache was cold or warm next to every figure.
- When a figure turns out wrong, correct it in place and say what changed, in neutral voice.

## Privacy

- No machine-local paths (home directories, scratch folders), hostnames, IP addresses other than
  `127.0.0.1`, internal tool names, or private data in any tracked file or commit message.

## Checks before committing

```bash
bash -n launch/*.sh
bash tier/tests/run_tests.sh
```
