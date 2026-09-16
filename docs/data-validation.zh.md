# 数据验证 —— arm 与参照 oracle 是怎么测量的

> 流水线的**第二阶段**。一个实例造好之后（见 **数据生成**），我们对它测量两类不同的东西：
>
> 1. **CA-agent 的 arm** —— 一个 frontier agent 在计量沙箱里，以 `train` 或 `no_train` 条件
>    真正去做这个任务。
> 2. **参照 oracle** —— 一组固定的"天花板"，说明*某一种知识最多能带你走多远*，好让任何 agent
>    的分数都有参照系。
>
> 先说清楚：这个系统里**没有任何统计 p 值**。实例级的量是 **π**（skeleton 占难度的份额，见
> 数据生成 §7）；每个 arm 和 oracle 报的都是准确率类指标（`overall`、`by_split`、`tail`、
> `headroom`）。代码在 `lumen1` 的 `~/code/Glyph-v2`；引用格式 `file:line`。

## 1. 指标

每个 oracle（和每个 arm）报的结构一样：

- **overall** —— 在被评分项上的平均**精确匹配**准确率。
- **by_split** —— 同上，但按测试档拆开：`iid`、`comp`、`depth`（数据生成 §6）。
- **tail** —— 只在 *tail* 项上的准确率（这些项需要的表 cell 从没被见过/买过）。没有 tail 项时为
  `None`。
- **headroom** —— 在 skeleton 天花板和 perfect 之间的归一化位置（`seal.py:15-33`）：

  ```
  headroom(score, ceiling) = (score - ceiling) / (1 - ceiling)      # ceiling >= 1 时为 None
  ```

  `0.0` = "不比'知道全部结构规则、但一个表条目都不知道'更好"；`1.0` = 完美；**负值是有意义的、
  不做裁剪**（你比纯 skeleton 基线还差）。

arm 和 oracle 都用 `.strip()` 后的**精确字符串匹配**评分（`answers.py:163`、`instance.py:395-404`）。
宽松的 digit-match 打分器**只**用于 π 的测量，这里从不用（`measure.py:44-46`）。

## 2. 评的是什么 —— 配对子集 vs 全量 test

刻意用两个不同的被评分总体：

- **参照 oracle** 评的是**按实例的配对子集**、**500 项**：`paired_subset(inst, n=500, seed=777)`
  （`subset.py:20-35`）。它是**分层的**（每档配额 ∝ 它在 test 中的占比）且**种子固定**，所以
  *同一个实例上的每个 oracle 都在同一批 500 项上评分*——这是"配对"对比，控制掉了项目难度。每个
  实例从**自己**的 test 项里抽**自己**的 500，子集**不跨实例共用**。（固定 500 让昂贵的 oracle
  ——GPU/API——负担得起，同时保持代表性。）
- **CA-agent 的 arm** 评的是**全量** `inst.test`——三档共 10000 项（`report.py:36`）。更大的
  总体带来约 0.5% 的标准误，才能在 1–2% 的水平上区分不同 arm。

## 3. 参照 oracle 电池

共五个 oracle。三个便宜（CPU，默认计算并提交），两个延后（GPU/API）。

### 3.1 skeleton 天花板（便宜）

**设置**：*真* skeleton 组合在 `IdentityTables`（`u(x)=x`, `b(x,y)=x`）之上
（`instance.py:350-382`）。"每一条结构规则都有，但一个表条目都没有。"它衡量**只靠结构/推理这
一半**能走多远。skeleton 天花板高 ⇒ 结构主导（高 π）的实例。它也是 `headroom` 的基线。

### 3.2 table 天花板（便宜）

**设置**：*真* 表组合在**平凡 skeleton** 之上（`semantics.py:168-186`：`UL`→普通 map、
`KL`→take-k、其余 identity）。故意**不是**空操作 skeleton——空操作会连带把表也悄悄消融（永不
调用原子算子），使测量有偏。它衡量**只靠表/记忆这一半**能走多远。table 天花板高 ⇒ 表主导
（低 π）的实例。

### 3.3 perfect（便宜）

**设置**：真 skeleton + 真表 ⇒ 按构造 `overall = 1.0`（`subset.py:63-68`）。用来 sanity check
实例可解、评测管线接对了。

### 3.4 weights 天花板 —— `[GPU]`，延后

**回答的问题**：在给定覆盖率下，把表放进*权重*（训练）能走多远？

**设置**（`weights_ceiling.py`）：在表的一个 `seen_frac` 切片上微调一个小 student 模型
（`Qwen/Qwen3-1.7B`、`lr=1e-4`、`steps=6000`、`batch=128`，`weights_ceiling.py:156`）。"seen"
集合用哈希选取，使其*不是*等差数列（否则会是人为偏易的区域）：`is_seen(key) =
(key·2654435761) % 100000 < frac·100000`，其中 `seen_u(i)` 以输入值为键、`seen_b(i,j)` 以
`i·7919+j` 为键（`weights_ceiling.py:41-50`）。

**数据粒度 —— 原子 cell。** 每条训练样本就是一次查表：`"u0 <值> ="` → `"<输出>"`（binary 为
`"b0 <值> <值> ="`）（`weights_ceiling.py:53-58`）。每个 seen 的 cell 一条，覆盖全部 3 个 unary
+ 2 个 binary 算子。student 记住原子 cell；值嵌入里的数字结构让它能外推到没见过的 cell。

**评分 —— 只测表。** `score_ceiling` 用**真** skeleton 组合在 *student* 学到的表之上
（`weights_ceiling.py:258-282`）。所以组合是精确给定的；这个 oracle 只隔离"student 能从
`seen_frac` 切片里把表学多好"。它还报 `entries_needed`（这些项需要的 distinct unary/binary cell 数）。

### 3.5 A0′ 天花板 —— `[API]`，延后

**回答的问题**：在给定覆盖率下，把表放进*上下文*（检索/in-context 证据）能走多远？

**设置**（`a0prime.py`）：`buy_evidence(inst, entries_target, probe_frac=0.7)` 通过发查询、分两
阶段来"买"事实（`a0prime.py:36-102`）：

- **probe** —— 浅、单算子（`depth_stop_prob=0.9`，budget 1），使每个答案孤立出一个表条目——就像
  agent 直接读表那样。
- **in-domain** —— 按 `demo_max_depth` 抽，和测试项同分布，使 skeleton 可从同一分布被推断。

停止条件数的是**去重后暴露的 unary 条目数**（`len(query_log.unary)`），不是原始事实条数。
`entries_target = frac · n_values`（即以"一张 unary 表的 cell 量"为分母）。

**数据粒度 —— 完整表达式。** 交给模型的 `facts` 是实际买到的 `(渲染表达式, 答案)` 对——probe
按设计接近原子，但 in-domain 事实是*带最终答案的完整组合表达式*。cell 数只是覆盖/停止统计；
上下文**不是**逐 cell 的表格转储。

**评分 —— 端到端。** `run_a0prime(inst, items, evidence, answer_fn)` 把证据放进 frontier 的上下文，
让它直接回答**测试项**（`a0prime.py:146-198`）。所以 frontier 既要在上下文里推表、又要自己做
skeleton 组合。（真正的 frontier 调用是后续的 `[API]` 步骤；模块本身从不发网络请求——`answer_fn`
是可注入的接缝。）`retrieval_split`（`a0prime.py:105-120`）把项分成 **covered**（需要的 cell 全被
暴露）与 **uncovered**，用于"检索 vs 外推"分析。

### 3.6 粒度上的不对称（重要）

两个"部分覆盖"的 oracle **并不**对称——train arm 也不对称（§4.3）：

| | 知识放在 | 数据粒度 | 组合 |
|---|---|---|---|
| weights 天花板 | 模型权重 | **原子 cell**（`op 值 = 输出`） | 给定（真 skeleton 组合在 student 表上） |
| A0′ 天花板 | 上下文 | **完整表达式**（+ probe） | 由 frontier 在上下文里完成 |

要比较"同覆盖率下 上下文 vs 权重"，必须先把这些差异对齐：粒度；只测表 vs 端到端；以及
`entries_target` 的分母——A0′ 数的是跨算子的 distinct unary ≈ `frac·4913`，而 `seen_u` 是
每算子 ≈ `frac·3·4913`，差约 3×。这些就是尚待处理的 A0′ 标准化设计点，也是 `[API]` 真跑被
门控的原因。

### 3.7 结果存在哪

`tools/run_reference.py` 写入 `docs/benchmark/reference_ceilings.json`，以实例 id 为键。默认跑
`CHEAP_ORACLES = (skeleton, table, perfect)`；`weights`/`a0prime` 只有 `--only` 请求时才跑
（`run_reference.py:37, 193`）。便宜的 oracle 存 `{overall, by_split, tail, headroom}`；
`weights`/`a0prime` 存同样结构但**以 `seen_frac` 为键**（默认扫 `[0.02, 0.05, 0.10]`，
`run_reference.py:196`），且证据在递增覆盖率上是累积的。

## 4. CA-agent 的 arm

### 4.1 RunConfig

一次 run 由 `RunConfig` 配置（`harness.py:197-216`）：

| 字段 | 默认 | 含义 |
|---|---|---|
| `arm` | *(必填)* | `"train"` 或 `"no_train"` |
| `preset` | `"pi_mid"` | 从哪个 preset 生成（设了 `instance_id` 则忽略） |
| `instance_seed` | `1001` | `generate()` 的种子 |
| `instance_id` | `None` | 设了就用固化 manifest 实例（校验 fingerprint） |
| `model` | `"claude-opus-4-8"` | frontier 模型（网关处锁定） |
| `student_model` | `"Qwen/Qwen3-1.7B"` | train arm 的本地 student（与 `model` 不同） |
| `q_cap` | `1000` | 总查询预算 |
| `submit_cap` | `20` | 强制切到 final 前的验证提交次数 |
| `tp` | `100` | practice 阶段轮数上限 |
| `tf` | `30` | final 阶段轮数上限 |
| `usd_line` | `300.0` | 累计花费安全线 |
| `n_val` | `5000` | 验证档大小 |
| `max_turns` | `200` | SDK 自身 agent 循环兜底（必须高于 `tp + tf`） |
| `effort` | `"high"` | 为可比性锁定 |

### 4.2 `train` vs `no_train`

**唯一**的区别是三个额外工具，只在 `train` 下可用（`session.py:16-41`）：

- `build_dataset` —— 把买到的/demo/验证查询对变成一个训练 JSONL。
- `train` —— 在数据集上全量微调本地 `student_model`（GPU 门控，受累计 GPU 秒预算约束；默认
  每次 1800 秒、总计 7200 秒）。
- `student_infer` —— 用得到的 checkpoint 去回答查表。

两个 arm 都能 `query` oracle、对验证 `submit`/`check_answers`、`final_answer`。`no_train` 完全
没有 student。`harness.run()` 只在 `arm == "train"` 时才建 `StudentPool`（`harness.py:330-331`、
`student.py`）。

### 4.3 train arm 的 student 到底训在什么上 —— 表达式，不是 cell

这一点微妙但重要。`build_dataset` 工具读 agent 写的 JSONL（`{"expr":…, "answer":…}` 行），
打包成 `Example(prompt="<expr> =", answer=" <answer>")`（`student.py:162-206`）。**train arm 的
student 是训在完整的 表达式→答案 对上的**，用的是 agent 自己选的表达式（demos、买来的 probe、
或自己生成的）——**没有**原子 cell 的入口。

这和 **weights 天花板** oracle（§3.4）是*不同、更粗*的粒度——后者枚举原子 cell。所以 train arm
和 weights 天花板**并没有**用同一种数据训 student，不能直接当作"同样的知识放进权重"来比较。

*（在 pilot `pi_mid`/种子 1001 的 train run 里观察到：agent 最大的训练集是 160 条 = 30 demos +
130 条自生成表达式；它还两头下注，手写了一个求解 skeleton 的符号求解器。它的 `overall` 是
0.4197，而 `no_train` 是 0.402——在这个接近交叉点的 π=0.44 实例上，训练几乎没起作用。）*

### 4.4 一次 run 怎么评分

`build_report`（`report.py:33-122`）把**已提交的最终答案**对**全量** `inst.test`（全部 1 万项）
做精确匹配评分，报告：

- `overall`、`by_split`（按测试档）、`by_depth`（按表达式树深度）；
- `tail` —— 同上但只在 tail 项；
- 每档及 tail 的 `headroom`，对 `inst.ceilings(items)`；
- 一个 `instance` 块（`seed, preset, pi, n_structural, atomic_ratio, uses_binary_tables`）；
- 一个 `validation` 块（提交历史、best/last、`gap_last_vs_test_iid`）；
- `covariates`（`q_used`、提交数、轮数、`val_lookup_solvable`、`test_covered`、
  `final_from_student`、`final_commit`、`run_status`）。

若没有合法提交，分数全为 0（`_zero_score`）。

### 4.5 practice 与 final 阶段

- **practice**（`drive_practice`，`harness.py:111-144`）—— agent 探索：可以 `query`、对验证
  `submit`、`check_answers`，以及（train arm）建/训/推理 student。结束于：第 `submit_cap` 次提交、
  agent 调用 `finish_practice`、`tp` 轮数上限、`usd_line`、或 3 轮空转。
- **final**（`drive_final`，`harness.py:146-191`）—— agent 通过 `final_answer` 提交答案。若结束
  时没提交但有一个通过的 `check_answers(set="test")` 路径，就自动提交那个路径
  （`final_commit="auto_checked_path"`）；否则分数保持为 0。

### 4.6 计量网关与查询 oracle

一个 `Gateway`（`gateway.py`）坐在被沙箱化的 Claude Code CLI 和 Vertex AI 之间：只转发**锁定**
模型的请求，并把每个响应的 `usage` 解析进 `Ledger`，使每个成功请求恰好计费一次。查询 oracle
`t_query`（`tools.py:79-106`）执行预算：一个查询可能被 OOD 策略*拒绝*（不计费）或撞上
`q_exhausted`（不计费），否则**先计费**——连格式错误的表达式也要花一次查询。`q_cap`（1000）
限查询数、`submit_cap`（20）限验证提交数、`usd_line`（$300）是每轮都检查的累计花费安全线。

## 5. 在 viewer 里看

run viewer 会为每个 run 叠加它所用固化实例的参照电池（`skeleton`/`table`/`perfect`，以及一旦
填充后按 `seen_frac` 的 `weights`/`a0prime`），显示每个 oracle 的 `overall` 和 `tail`，外加
`by_split`（`iid · comp · depth`）和 `headroom`。Instances 视图让你按 档 → id 浏览这 15 个固化
实例，看同样的电池。

---

*配套：**数据生成** —— 一个实例是怎么造出来的。*
