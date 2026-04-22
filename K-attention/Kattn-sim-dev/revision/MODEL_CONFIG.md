# KNET 模型配置规范

> **必读**：每次新建实验脚本前对照此表，避免模型配置错误导致实验结果偏低。

---

## 正确配置一览

| model_type | 任务 | 模型类 | reverse | band mask | 推荐 lr |
|------------|------|--------|---------|-----------|---------|
| `KNET_rc` | RC | `KattentionModel` | **True** | ❌ | **1e-2** |
| `KNET_uncons_rc` | RC（消融） | `KattentionModel_uncons` | **True** | ❌ | **1e-2** |
| `KNET` | Markov | `KattentionModel_mask` | False | ✅ ±2 diagonal | **1e-2** |
| `KNET_uncons` | Markov（消融） | `KattentionModel_uncons_mask` | False | ✅ ±2 diagonal | **1e-2** |

---

## 关键说明

### reverse=True（RC 任务）
- `KattentionV4` 中 `reverse=True` 时，Key 从**翻转后的序列** `X.flip([1])` 计算
- Query 来自正链，Key 来自反链 → 注意力矩阵捕获正链和反链之间的互补关系
- 这是 RC 任务中 KNET 性能达到 ~1.0 的关键，去掉 reverse 后 AUROC 仅 ~0.91

### band mask（Markov 任务）
- `KattentionModel_mask` 在 forward 中将 attn_logits 乘以 ±2 对角线掩码
- 只保留 `|Q_pos - K_pos| <= 2` 的注意力值，对应一阶 Markov 先验（相邻依赖）
- RC 任务**不需要** band mask（reverse=True 已提供足够的归纳偏置）

### 学习率
- **所有 KNET 变体统一用 1e-2**
- 1e-4 收敛不足，AUROC 仅约 0.8；先测 1e-2，收敛有问题再降

---

## run_bmk.py 中的实现（`kattn_version` 控制 reverse）

```python
# RC 任务：reverse=True 通过 kattn_version="v4_rev" 传入
KattentionModel(kattn_version="v4_rev", ...)          # KNET_rc
KattentionModel_uncons(kattn_version="v4_rev", ...)   # KNET_uncons_rc

# Markov 任务：reverse=False，加 band mask
KattentionModel_mask(kattn_version="v4_mask", ...)    # KNET
KattentionModel_uncons_mask(kattn_version="v4", ...)  # KNET_uncons
```

`KattentionV4.__init__` 中：`reverse = "rev" in kattn_version`

---

## 历史错误记录

| 时间 | 错误 | 后果 |
|------|------|------|
| 2026-04 Exp B | `KNET_rc` 误用 `kattn_version="v4"`（reverse=False）| RC AUROC 最高仅 0.916（预期 ~1.0） |

**已修复**：2026-04-22，`run_bmk.py` 中 `KNET_rc` 和 `KNET_uncons_rc` 改为 `kattn_version="v4_rev"`。  
**待操作**：重新运行 Exp B 中 KNET_rc 的 15 条 RC runs。
