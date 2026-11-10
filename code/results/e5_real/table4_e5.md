### Table 4 · E5 阈值搬运

| setting | n_images | n_meas | pfa_transport | pfa_recal | pfa_no_convert | alpha | drift_transport | drift_recal | drift_no_convert | insufficient | pass | better_than_no_convert |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP16→INT8 | 232 | 949703 | 0.00111298 | 0.00124144 | 0.355071 | 0.001 | 0.0464872 | 0.093926 | 2.55032 | False | True | True |
| 重标定上界 | 232 | 949703 |  | 0.00124144 |  | 0.001 |  | 0.093926 |  | False |  |  |
