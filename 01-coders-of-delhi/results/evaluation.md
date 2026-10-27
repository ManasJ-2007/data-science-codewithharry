# Evaluation (3 seeds, 1000 users, 6000 friendships, 64 pages)

## People You May Know (600 hidden friendships per seed)
| Method | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| mutual | 0.062 ± 0.007 | 0.208 ± 0.008 | 0.320 ± 0.009 | 0.146 ± 0.006 |
| jaccard | 0.071 ± 0.011 | 0.228 ± 0.009 | 0.338 ± 0.005 | 0.159 ± 0.009 |
| **adamic_adar** | 0.071 ± 0.010 | 0.229 ± 0.008 | 0.357 ± 0.014 | 0.160 ± 0.008 |
| popularity | 0.001 ± 0.001 | 0.009 ± 0.002 | 0.017 ± 0.002 | 0.010 ± 0.001 |
| random | 0.001 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.000 |

## Pages You Might Like (920 hidden likes per seed)
| Method | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| overlap | 0.317 ± 0.024 | 0.637 ± 0.006 | 0.753 ± 0.008 | 0.458 ± 0.015 |
| jaccard | 0.319 ± 0.024 | 0.640 ± 0.009 | 0.759 ± 0.009 | 0.461 ± 0.015 |
| **friends** | 0.328 ± 0.003 | 0.650 ± 0.005 | 0.752 ± 0.011 | 0.468 ± 0.003 |
| hybrid | 0.323 ± 0.022 | 0.644 ± 0.009 | 0.766 ± 0.007 | 0.466 ± 0.014 |
| popularity | 0.035 ± 0.008 | 0.190 ± 0.004 | 0.347 ± 0.002 | 0.133 ± 0.006 |
| random | 0.017 ± 0.000 | 0.083 ± 0.000 | 0.167 ± 0.000 | 0.078 ± 0.000 |

## Cleaning (10% of users corrupted)
| Seed | Issues fixed | Users in -> out | Integrity violations after |
|---|---|---|---|
| 0 | 399 | 1020 -> 976 | 0 |
| 1 | 423 | 1020 -> 975 | 0 |
| 2 | 357 | 1020 -> 981 | 0 |

## Speed (pure Python, both recommenders, every user)
| Users | ms per user |
|---|---|
| 1,000 | 0.41 |
| 5,000 | 1.82 |
| 10,000 | 3.56 |
