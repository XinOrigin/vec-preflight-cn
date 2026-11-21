# Virtual Embryo Challenge (NeurIPS 2026) 中文实战与避坑指南

> 本文档依据官方规则页（`virtualembryo.ai/challenge/rules`、`evaluation`、`baselines`、`timeline`）以及官方离线评分源码 [`aristoteleo/veckit`](https://github.com/aristoteleo/veckit) 逐行核对整理，面向中文参赛者与 Agent Team 选手。

---

## 1. 赛事总览与关键时间节点（P3 终审阶段）

Virtual Embryo Challenge 由斯坦福大学 Qiu Lab 主办（Laude Institute Moonshots Seed Grant 资助），作为 **NeurIPS 2026 官方竞赛** 举办，总奖池 **$112K**：

| 奖项 | 金额 | 名额与规则 |
|---|---|---|
| Human Team 赛道 | $27K | 一等奖 1×$8K、二等奖 2×$5K、三等奖 3×$3K |
| Agent Team 赛道 | $27K | 一等奖 1×$8K、二等奖 2×$5K、三等奖 3×$3K（需提交可核验 Agent 证据） |
| Generality Award | $8K | 跨赛道单奖，奖励用同一共享架构同时解决多任务的最佳方案（可叠加名次奖） |
| Travel Awards | $30K | 15–20 份 NeurIPS 参会资助 |
| Community Contribution Award | $20K | 滚动审核至 2026-12-11，单条最高 $200，可多拿，不要求参赛或上榜 |

**P3 阶段关键规则（务必在提交前确认）：**
- **最终提交截止**：**2026-11-25 23:59 AoE**（北京时间 **2026-11-26 20:00**；官方于 11-16 提前了一周）。
- **额度限制**：每个可用榜单在 P3 阶段最多仅 **2 次最终提交**。
- **P2 验证集真值解锁警告**：P3 允许申请下载 P2 验证集真值用于模型重训，但**一旦领取验证集真值或提交首次 P3 正式结果，全队将永久失去继续提交 P2 榜的资格**（领取真值本身不消耗 P3 的 2 次提交额度）。
- **过线门槛（Floor Gate）**：所有赛道只有最终分数**严格高于 Floor（每榜 50 分，三任务总分 > 150）**才具备获奖资格。

---

## 2. 三大任务（T1 / T2 / T3）数据结构与评分口径

基于 `aristoteleo/veckit` (`score_h5ad.py` 与 `T1/T2/T3/metrics_v2.py`) 核对：

| 任务 | 生物问题 | 基因面板 | 必需字段 | Floor 基准 |
|---|---|---|---|---|
| **T1 (Temporal)** | 单细胞 RNA 时间外推（训练 E8.5/E9.5 → 验证 E10.5 → 测试 E12.5） | **32,285 全转录组基因** | `X`（2D 非负 `log1p` float32） | `copy_last` |
| **T2 (Spatial-temporal)** | 3D 空间转录组时空预测（MERFISH，含心脏与全胚胎） | **500 MERFISH 基因** | `X` + **`obsm['spatial_3D']`** (`N×≥3` 有限 float32) | `copy_last` |
| **T3 (Perturbation)** | 基因敲除突变体预测（表达 + 3D 形态） | **500 MERFISH 基因** | `X` + **`obsm['spatial_3D']`** | `wt_identity` |

### 官方 0–100 标度换算公式（Floor = 50，Ceiling = 100）

官方榜单不直接把不同量纲的原始指标相加，而是先用两个锚点归一化：
- **Floor ($m_{\text{floor}}$)**：`copy_last`（T1/T2）或 `wt_identity`（T3）在同一拆分上的得分，映射后**恒等于 50 分**。
- **Ceiling ($m_{\text{ceiling}}$)**：将真值细胞随机拆分为两个互不相交的半集，用一半评估另一半得到的可达上限，映射后**恒等于 100 分**。

设单指标距离上限的差距为 $d(m)$（越小越优取 $m - m_{\text{ceiling}}$，越大越优取 $m_{\text{ceiling}} - m$；带符号的对数比值指标 `scale_log_ratio`、`count_log_ratio` 先取 $|m|$ 再按越小越优处理），设 $d_{\text{floor}} = |m_{\text{ceiling}} - m_{\text{floor}}|$，则：

$$\text{skill}(m) = \min\!\left(\frac{d_{\text{floor}}}{d_{\text{floor}} + d(m)},\; 1\right) \times 100$$

- 当 $m = m_{\text{floor}}$ 时，$\text{skill} = 50$；
- 优于上限时截断为 $100$；
- 缺失或返回 `NaN` 的指标记为 $0$（绝不静默跳过）。

T1 四个维度的官方权重：`DE gene recovery (25%)`、`Change direction (25%)`、`Cell-state distribution (30%)`、`Gene-gene co-variation (20%)`。

---

## 3. 必看 8 大提交避坑清单（全部对照 `veckit` 源码证实）

1. **评分器绝不自动归一化（No Auto-Renormalization）**
   `score_h5ad.py` 直接读取 `adata.X` 并强制要求非负（`X < 0` 直接报错）。如果提交了原始 count 整数矩阵（甚至 `veckit/data/sample_heart_9.25.h5ad` 本身最大值达 `63.0`，而 `sample_heart_9.5.h5ad` 是 `log1p` 后的 `8.43`），不会报错退出，但 `pb_rel_err` 和 `library_size_ratio` 会直接炸掉。提交前务必使用 `vec-preflight check --reference <ref.h5ad>` 核查尺度。
2. **T2/T3 空间坐标键名必须是 `obsm['spatial_3D']`**
   不是 `obsm['spatial']`，也不是 `obsm['spatial_2D']`！`score_h5ad.py` 硬编码检查 `"spatial_3D" in a.obsm` 且要求 `shape == (n_obs, >=3)` 且无 `NaN/Inf`。
3. **不要只预测"平均细胞"（Mean Collapse 陷阱）**
   如果所有生成细胞高度相似，虽然 `pseudobulk_pearson` 看起来极高（0.88–0.95），但真正计分的 `mmd_u`、`energy_distance`、`variogram` 与约束项 `variance_ratio` 会严重失分。
4. **基因面板名称与列顺序必须完全一致**
   提交 `.h5ad` 的 `var_names` 必须与任务参考集逐一对应且顺序一致（`adata[:, ref.var_names].copy()`）。注意：`T1/metrics_v2.py` 旧注释里写了 `"full-500-gene-space"`，但官方已在 [veckit#3](https://github.com/aristoteleo/veckit/issues/3) 确认那是过时注释——T1 实际使用完整 **32,285** 参考基因面板，T2/T3 使用 **500** 基因面板。
5. **本地跑 `veckit` 时务必显式传 `--reference`（T1/T2）或 `--wt`（T3）**
   如果省略 `--reference`/`--wt`，`veckit` 会默认用 `--input` 自己充当参考阶段，导致 `de_score`、`de_direction`、`severity_slope` 全部输出 `0.0`（只适合测 `copy_last`/`wt_identity` 基线）。
6. **`obs['celltype']` 不参与排名，不要花时间伪造标签**
   官方已移除依赖提交方细胞标签的指标，`composition_JSD` 由官方在真值上训练的冻结分类探针（frozen probe）统一推断。
7. **T1/T2 的主要变化不是"老细胞平移"，而是新细胞类型分化**
   E8.5 与 E9.5 仅共享 5 种细胞类型，后续阶段约 2/3 细胞为新分化类型；仅做 `pseudobulk_shift` 往往难以捕获新群落分布。
8. **Track 2 (Agent Team) 必须附带 ≥2 类独立证据**
   官方规则 §8/§14 规定：Agent Team 仅允许人类写初始 prompt，过程中禁止人类看中间分数干预调参；提交时必须附带至少 **2 类**（推荐全部 3 类）来自 `{trajectory, prompts, harness}` 的真实证据文件，否则提交将被扣留不进入评分队列，更无法领取奖金。上传前建议用 `vec-preflight check-evidence --dir <dir>` 自动检查 JSON/JSONL 完整性并扫描误留的 API Key。

---

## 4. 一键本地体检与打分示例

```bash
# 1) 检查 .h5ad 格式、非负性、尺度、基因面板与 3D 坐标
vec-preflight check --task T2 --input pred.h5ad --reference sample_heart_9.5.h5ad

# 2) 调用 veckit 并自动计算 floor=50 / ceiling=100 标度分
vec-preflight score \
  --task T1 \
  --input pred.h5ad \
  --target sample_9.5.h5ad \
  --reference sample_8.5.h5ad

# 3) 检查 Agent Team 证据目录（轨迹 / Prompt / Harness + 密钥泄露扫描）
vec-preflight check-evidence --dir ./agent_evidence
```
