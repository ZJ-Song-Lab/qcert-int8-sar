### Table 6 · E8 检测兼容性与对照

| method | source | pd_at_1e-3 | pd_at_1e-4 | map50 | pfa | pfa_std_across_buckets | R | latency_ms | n_box | note |
|---|---|---|---|---|---|---|---|---|---|---|
| CA-CFAR（像素域） | dump | 0.139194 | 0.0750916 | nan | 0.000294971 | nan | nan | nan | 546 |  |
| YOLO11n + 全局阈值 | dump | 0.994505 | 0.987179 | 98.27 | 0.00103749 | nan | nan | 45.5 | 546 |  |
| YOLO11n + QCert(FP16) | dump | 0.990842 | nan | 98.27 | 0.00174112 | nan | 1 | 45.5 | 546 |  |
| YOLO11n + QCert(INT8) | dump | 0.990842 | 0.985348 | 97.98 | 0.00174112 | nan | 1 | 34.68 | 546 |  |
| QCert（完整） | dump | 0.990842 | 0.985348 | 97.98 | 0.00174112 | nan | 1 | 34.68 | 546 | 逐场景整数分位 + 除法消除 |
