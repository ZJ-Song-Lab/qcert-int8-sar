# Table 4 · E5 阈值搬运（压测集 300 张，logit 分数场，alpha=1e-3）

| 设置 | n_img | n_meas | P_fa(搬) | P_fa(重标) | P_fa(不换) | d(搬) | d(重标) | d(不换) | pass | 优于不换 |
|---|---|---|---|---|---|---|---|---|---|---|
| 设置1 同传感器换构建(calmA->roughB) | 300 | 1230836 | 0.001236 | 0.001183 | 0.0002965 | 0.09193 | 0.07296 | 0.5279 | True | True |
| 设置2 FP16->INT8(基准) | 300 | 1230836 | 0.001236 | 0.001183 | 0 | 0.09193 | 0.07296 | inf | True | True |

- 搬运公式: lambda_int_new = round(lambda_int_old * (dSig_new/dSig_old) * (dS_old/dS_new))
- 验收(04 §8): d(搬) <= 1.5 * d(重标)，且 d(搬) <= d(不换)
- 设置1: delta_s 0.08399->0.09144（calmA->roughB, cv3 真实输出步长）
- 设置2: delta_s 0.0078->0.09144（FP16->INT8 基准）
- 两设置 P_fa(搬)/P_fa(重标) 相同：z_new 仅由 new.delta_sigma=0.0466 决定，搬运保 lambda_float
- 设置2 P_fa(不换)=0：FP16 标定的 lambda_int 不换算直接用于 INT8 -> lambda_float 放大 ~23x -> 阈值完全失效（检测归零）
- 设置3（跨传感器 S1->GF-3）：无第二传感器数据，留作未来工作