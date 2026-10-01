# I5 — reversible memory branches (D09/D10)

- D09 isolation: claim visible before apply **True**, review gate `INVALID_TRANSITION`, applied → `archived`, recall dropped **True**, replay `True` → met **True**
- moved parent: submit `STALE_PROPOSAL`, apply `STALE_PROPOSAL` → met **True**
- D10 suppressed parent: apply `NOT_FOUND_OR_UNAUTHORIZED`, branch stays `live` → met **True**
- D10 purged parent: apply `INVALID_TRANSITION`, revisions ['active', 'erased'], branch `erased`, no receipt **True** → met **True**
- held→rebase: held in tx `held`, apply `INVALID_TRANSITION`, rebase `live` → met **True**
- abandon/diff: diff moved, apply `INVALID_TRANSITION`, disposition `archived` → met **True**

**met: True**
