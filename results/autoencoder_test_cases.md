# Autoencoder novelty evaluation
Repo: `E:\Documents\Financial-Market-Risk-Analysis` (51 functions) | cases: test_cases | bottleneck: 16

- Mean reconstruction error on training functions: 0.00043
- Mean error on held-out folds (normal, cross-validated): 0.00195
- Mean error on the test files: 0.00232

| File | Violations | Error | Novelty (0-1) |
|---|---|---|---|
| conformant_1.py | - | 0.00252 | 0.78 |
| conformant_2.py | - | 0.00181 | 0.57 |
| conformant_3.py | - | 0.00234 | 0.76 |
| conformant_4.py | - | 0.00259 | 0.80 |
| conformant_5.py | - | 0.00099 | 0.12 |
| deviant_1.py | function_naming, param_naming | 0.00295 | 0.86 |
| deviant_2.py | function_naming | 0.00284 | 0.86 |
| deviant_3.py | param_naming | 0.00264 | 0.80 |
| deviant_4.py | vague_params | 0.00213 | 0.63 |
| deviant_5.py | raises_exception | 0.00219 | 0.65 |
| deviant_6.py | bare_except | 0.00256 | 0.78 |
| deviant_7.py | logging_module | 0.00210 | 0.61 |
| deviant_8.py | string_format | 0.00228 | 0.71 |
| deviant_9.py | import_star | 0.00144 | 0.41 |
| deviant_10.py | function_naming, param_naming, vague_params, raises_exception | 0.00338 | 0.88 |

Mean novelty: conformant 0.61, deviant 0.72
AUC deviant-vs-conformant: 0.64  (0.50 = the score cannot tell them apart)

Off-topic control: 17 functions from `E:\Documents\MediSage`
Mean novelty: foreign 0.95 vs the repo's own held-out folds 0.49
AUC foreign-vs-own: 0.96
