# I2 — repair-local recomputation vs full rebuild (D03/D04)

- corpus: 16 claim slots, 16 persisted views, 1 localized correction
- objects scanned: repair **2** vs rebuild **65** → 96.9% less
- objects evaluated: **2** vs **32** → 93.8% less
- objects recomputed: **3** vs **18** → 83.3% less
- post-state equal (corrected value live on both arms): **True**
- held/stale marks inside the correction commit: {'views': 1, 'observations': 1, 'branches': 0}
- D03 met (≥50% on evaluated+recomputed): **True**
- D04 incomplete-forecast refused: True (CONTEXT_INCOMPLETE)
- D04 grown-closure refused: True (CONTEXT_INCOMPLETE), partial writes 0
- met: **True**

| arm | scanned | evaluated | recomputed |
| --- | --- | --- | --- |
| repair | 2 | 2 | 3 |
| full rebuild | 65 | 32 | 18 |
