# Dataset Redaction — `issues11k.csv`

**Date:** 2026-07-30
**Scope:** 1 row of 6,600. 8 substring replacements. No rows added, removed, or relabeled.

## What happened

`issues11k.csv` was previously gitignored, which meant a fresh clone could not
bootstrap the pipeline — the train/test splits and FAISS indices regenerate
deterministically, but the source pool does not. When the file was committed so
the repo would be self-contained, GitHub push protection (GH013) rejected the
push: the scraped issue text contains a credential matching the Bitbucket Server
Personal Access Token pattern.

The credential is not ours. It belongs to a third party who pasted their real
Flux/Bitbucket password into a public `kubernetes/kubernetes` bug report — the
issue is titled *"secret with space after `--from-literal=password='password'`
creates di..."*, and the whole point of the report is that `kubectl` mishandled
the quoting, so the reporter included the literal value to demonstrate it. It is
already public upstream, but committing it verbatim would make this repository
redistribute someone else's credential, so it was redacted instead.

## What was changed

Row index 5493 (0-based, over `issues11k.csv` as read by `csv.DictReader`),
`repo = kubernetes/kubernetes`. Only the `body` field was touched.

| Original string | Replacement | Count |
|---|---|---|
| `MTQyMzM1…5fj5` (raw credential) | `REDACTED_SECRET` | 5 |
| `TVRReU…ajU=` (its base64 encoding) | `REDACTED_SECRET_BASE64` | 2 |
| `TVRReU…ajWg` (base64 variant in the report) | `REDACTED_SECRET_BASE64` | 1 |

The associated `username: c2FtaXIucGF0cnk=` base64 was left intact — it encodes
only a username, not a credential, and GitHub's scanner did not flag it.

`issues11k_train.csv` was updated with the identical substitutions so the local
splits stay byte-consistent with the source pool; the affected row falls in the
train split, so `issues11k_test.csv` needed no change. Both split files remain
gitignored and regenerate from the redacted source.

## Effect on results

None that is meaningful. The change alters ~120 characters inside one issue body
in the training pool. It does not affect any label, any split boundary, or any
of the 3,300 test issues. All published metrics were computed before the
redaction and are unaffected; re-running the pipeline from the redacted source
can perturb at most the retrieval neighborhood of that single training issue.

## Reproducing

The redaction is a pure substring replacement. Applying the three substitutions
in the table above to a fresh scrape reproduces the committed file exactly.
