# Feature engineering

The Rule view is **not** a separate module: it is built inside
`catboost_backbone/g2_common.py::build_views` (search for `rules.append`).

Four binary indicators, no fitted parameters:

```python
coat = PIPE_COATING_TYPE.upper() in {'NONE', 'UNKNOWN', 'N/A'}
age  = PIPE_AGE_AT_ACCIDENT <= 10
mfg  = MANUFACTURE_AGE_AT_ACCIDENT <= 5
pr   = PRESSURE_RATIO > 1.2
```

A fifth rule (`prior`, on `PRIOR_DAMAGE`) is present in the code but never generated for
`MAIN_13`, because `PRIOR_DAMAGE` is one of the five excluded label-adjacent fields.
So the rule view is exactly 4-dimensional.

The Neural view is `h = W_neural(P_CB)`, i.e. a learned linear projection of the six
CatBoost class probabilities — not an independent feature extractor.

Numeric imputation (fit-split median) and categorical tokenisation (`NOT_RECORDED`) also
live in `g2_common.py`.
