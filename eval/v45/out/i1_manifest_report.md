# I1 — counterevidence-first manifest ablation (D01/D02)

- sample: 8 topics × 2 arms (16 runs), max_items=2
- false-current (top-k): **8**
- false-current (manifest): **0**
- manifest contrary included: 8, contrary labeled: 0
- identifier hits top-k/manifest: 8/8
- tokens top-k/manifest: 2480/2464
- D02 fell: True, identifier kept: True → met: True

| topic | top-k fc | manifest fc | contrary refs (incl/omit) | labels |
| --- | --- | --- | --- | --- |
| alpha0 | True | False | 2/14 | - |
| alpha1 | True | False | 2/14 | - |
| alpha2 | True | False | 2/14 | - |
| alpha3 | True | False | 2/14 | - |
| alpha4 | True | False | 2/14 | - |
| alpha5 | True | False | 2/14 | - |
| alpha6 | True | False | 2/14 | - |
| alpha7 | True | False | 2/14 | - |
