# 数据生成 —— 一个 Glyph instance 是怎么造出来的

> 流水线的**第一阶段**。给定 `(seed, preset)`，`generate()` 确定性地产出一个自包含的任务
> （"instance"）：一个隐藏程序（skeleton + tables）、一组上下文示例（demos）、一个封存的
> 测试集、一个验证集。本文讲清楚值空间、配置旋钮、三个 preset、生成流程，以及实例难度 π
> 是怎么测的。
>
> 代码在 `lumen1` 的 `~/code/Glyph-v2`。引用格式为 `file:line`。

## 1. 什么是一个 instance

一个 instance 是一个定义在有限值空间上的隐藏函数 `P = skeleton ∘ tables`，外加对它探测和评分
所需的数据：

- **skeleton**：一个小的、**可描述**的程序（结构算子的组合），从一个**有限文法**里采样得到。
- **tables**：每个算子私有的、**不可描述**的查找函数（冻结的随机 MLP）。
- **demos**：30 条已解的 `(表达式, 答案)`，agent 在上下文里能看到。
- **test**：一个大的封存集合（默认 10000），分成 `iid` / `comp` / `depth` 三档。
- **val**：一个验证集（默认 5000），run 过程中 agent 可以对它做聚合打分。

把任务拆成 **skeleton** 和 **tables** 的全部意义在于：这两半的获取方式不同——skeleton 是可以
推断、可以写下来的"代码/推理"；tables 是只能靠看例子获得的"记忆"（详见配套的 **数据验证** 文档）。

## 2. 值空间与渲染

每个值是有限空间 `V` 中的一个点，大小为

```
|V| = base ** n_digits = 17 ** 3 = 4913     （config.py:47-48, 160-162）
```

一个值的下标被编码成 `n_digits` 位小端数字，每位在 `[0, base)` 内（`grammar.py:60-67`）。
展示给 agent 的**表面形式**由 `cfg.value_form`（默认 `"letter_sep"`）决定，把每位数字渲染成
字母 `a..q`（17 个符号）——例如某个值 → `v_k_e_e`（`grammar.py:69-81`）。`flat` 形式
（`v1234`）被**故意弃用**，因为它抹掉了 tables 赖以泛化的数字结构（`config.py:12-38`）。

**为什么是 3 位、base 17**：这样既得到 4913 个不同的值，又让每个值只占很短、固定长度的
token；更关键的是，让每个值都带有内部的**数字结构**，tables 才能在其上泛化（见 §3）。

## 3. 算子与表

**结构算子**（skeleton 的积木）：共定义 8 个，`s0..s7`（`grammar.py:28-38`）。每个有一个
*shape*，决定它返回值还是列表（`UL`/`LB`/`L`/`KL`）。`enabled_ops(cfg)` 取前 `n_structural`
个；`s0`（类 map）和 `s1`（类 fold）排在最前，因为它俩是真正**消费原子表算子**的两个
（`grammar.py:42-47`）。

**原子算子**（表）：默认 `n_unary = 3`（`u0, u1, u2`）、`n_binary = 2`（`b0, b1`）
（`grammar.py:49-54`）。

**表的大小**（`tables.py:6-8`）——表是**函数**，不是存下来的数组：

| 类型 | 签名 | 若列举成表的条目数 | 每个 instance |
|---|---|---|---|
| unary | `V → V` | ~4913 | × 3 个算子 ≈ 14,739 |
| binary | `V × V → V` | ~2400 万（`4913²`） | × 2 个算子 ≈ 4800 万 |

表是**冻结的随机 MLP**——`FrozenMLP`，两层 tanh，随机权重，从不训练（`tables.py:38-56`）。
两个性质同时成立（`tables.py:10-24`）：

- **structured（有结构）**：数字嵌入在所有值之间共享，数字相近的值输出也相近。这是学习者能
  **外推**到没见过的条目的**唯一**原因。
- **indescribable（不可描述）**：权重是随机的，"把函数写下来"就等于抄权重矩阵。你说不出规律，
  只能从例子里学。

binary 表以逐数字为主，加一个由 `binary_coupling`（默认 0.25）加权的弱全局 `mix` 项；纯逐数字
（`coupling = 0`）只有 `n_digits · base²` 个不同的"部件"——小到可枚举——所以这个耦合项才让
binary 真正变难（`tables.py:20-24, 89-100`）。`IdentityTables`（`u(x)=x`, `b(x,y)=x`）是
skeleton 天花板用的退化基线（`tables.py:261-274`）。

## 4. 配置 —— `GlyphConfig`

`GlyphConfig` 是一个 frozen dataclass（`config.py:44-186`）。全部字段：

| 字段 | 默认 | 含义 |
|---|---|---|
| `base` | 17 | 每位的进制 |
| `n_digits` | 3 | 数字位数（`base**n_digits = n_values`） |
| `d_digit` | 16 | 每位数字的嵌入维度 |
| `value_form` | `"letter_sep"` | 值的渲染形式（§2） |
| `n_structural` | 5 | 启用 `s0..s7` 中的几个 |
| `max_transform_depth` | 2 | `<transform> then <transform>` 的最大嵌套深度 |
| `guard_prob` | 0.5 | 结构算子带 `if/then/else` 守卫的概率 |
| `n_unary` | 3 | unary 原子算子个数 |
| `n_binary` | 2 | binary 原子算子个数 |
| `mlp_width` | 64 | 每个 `FrozenMLP` 的隐藏宽度 |
| `mlp_temp` | 1.0 | 激活前缩放（越大越不平滑） |
| `binary_coupling` | 0.25 | 0 = 纯逐数字；越大 binary 表越难 |
| `unary_coupling` | 0.25 | unary 上的弱全局耦合；`None` = 单个联合 MLP |
| `decode` | `"whiten"` | MLP 的实数输出如何变成合法符号 |
| `atomic_ratio` | 0.5 | **扫 π 的主旋钮**（§5） |
| `binary_freq` | 0.35 | 采样根节点想要值（fold）而非列表的概率 |
| `max_expr_depth` | 4 | 表达式最大嵌套深度 |
| `demo_max_depth` | 2 | demos / `iid` / `comp` 的深度上限 |
| `depth_stop_prob` | 0.15 | 某层递归提前停止的概率 |
| `list_len_range` | (2, 4) | 列表字面量的合法长度 |
| `n_demos` | 30 | 上下文示例对数 |
| `n_iid` | 6500 | `iid` 测试档大小 |
| `n_comp` | 2300 | `comp` 测试档大小 |
| `n_depth` | 1200 | `depth` 测试档大小 |
| `n_val` | 5000 | 验证档大小 |

派生量（`config.py:160-170`）：`n_values = base**n_digits`、`n_test = n_iid+n_comp+n_depth`
（默认 10000）。`with_(**kw)` 返回改动后的副本；`scaled(n_test)` 在保持 65/23/12 比例下
缩放测试集。

配置里**没有** `held_pairs` 或算子数量的字段——held pairs 是在生成时按实例计算的（§6），不是旋钮。

## 5. Preset 与 π 旋钮

内置三个 preset（`config.py:198-279`）；未列出的字段取 dataclass 默认值。

| preset | `atomic_ratio` | `n_structural` | `max_transform_depth` | `guard_prob` | `max_expr_depth` | `demo_max_depth` |
|---|---|---|---|---|---|---|
| `pi_low` | **0.85** | 5 | 0 | 0.0 | 3 | 2 |
| `pi_mid` | 0.5 | 5 | 2 | 0.5 | 4（默认） | 2（默认） |
| `pi_high` | **0.15** | 8 | 3 | 0.9 | 5 | 3 |

**`atomic_ratio` 是扫 π 的主连续旋钮**（`config.py:128`）。它让表达式采样偏向**消费表**的算子
还是纯结构算子（`instance.py:85-90`）：

- **低 `atomic_ratio` → 高 π**（`pi_high`，0.15）：表达式很少碰表，难度集中在 skeleton。
  8 个结构算子全开、深嵌套、带守卫。
- **高 `atomic_ratio` → 低 π**（`pi_low`，0.85）：表达式不停碰表，难度集中在 tables。
  结构算子近乎平凡、无守卫、无组合。

次级旋钮随 `atomic_ratio` 单调变化（从 `pi_high` 到 `pi_low`，skeleton 复杂度递减），强化同一根轴。

**重要 —— 名义 π vs 测得 π。** preset 的**名字**只承载**意图中的** π；真实值是那个具体
`(seed, preset)` 下 `measured_pi()` 报出来的，而且会**漂移**（一个 `pi_mid` 种子可能测进
`pi_high` 档）。相图坐标和固化 benchmark 一律用**测得**值，绝不用 preset 名字
（`config.py:190-192`）。这正是那 15 个固化实例按**测得 π** 而非 preset 挑选的原因（见 benchmark manifest）。

## 6. 生成流程

`generate(seed, cfg)` 就是构造 `GlyphInstance(cfg, seed)`（`instance.py:499-500`）。构造函数按
顺序跑下面这些步骤，全部走**同一个** RNG（`instance.py:216-236`）：

1. `rng = np.random.default_rng(seed)` —— 只创建**一次**，然后贯穿每一步。
2. `skeleton = sample_skeleton(cfg, rng)` —— 每个启用的结构算子采一个 `StructSem`，从有限的
   transform 文法里抽（`semantics.py:134-165`）。从**有限**文法采样，正是 skeleton 永远能用有限
   词句描述的保证。
3. `tables = Tables(cfg, rng)` —— 冻结 MLP（§3）。
4. `P = Interpreter(cfg, skeleton, tables)` —— 参考解释器，`P = skeleton ∘ tables`。
5. `held_pairs = _draw_held_pairs(cfg, rng)` —— 见下。
6. `demos = _make_demos(rng)` —— 30 条、深度 `demo_max_depth`、禁用 held pairs。
7. `test = _make_test(rng)` —— 三个档（见下）。
8. `val = _make_val(rng)` —— 5000 条，抽法同 demos，但与 demos+test 不相交。

因为整条流程都串在单个 `default_rng(seed)` 上（没有别的 RNG、没有全局随机、没有 torch），
同样的 `(seed, cfg)` 会**逐字节**复现同一个实例——这就是 fingerprint 和 `load_instance` 的
根基（§8）。

### 测试档定义（`instance.py:432-435`）

| 档 | 大小 | 深度预算 | held pairs | 额外 |
|---|---|---|---|---|
| `iid` | 6500 | `demo_max_depth` | 禁用 | 普通同分布 |
| `comp` | 2300 | `demo_max_depth` | **必须出现** | 测某个 held-out 的算子**组合**能否泛化 |
| `depth` | 1200 | `max_expr_depth` | 禁用 | `min_depth = demo_max_depth + 1`——严格**比任何 demo 更深** |

- **iid**：约束同 demos；普通项。
- **comp**：同样的浅深度，但每一项都**必须包含**一个 held-out 的（外层, 内层）算子对——某个
  组合能否泛化？
- **depth**：允许一直嵌套到 `max_expr_depth`，且被强制比任何演示过的都**更深**——对没见过的
  深度能否泛化？

每个 `TestItem` 记录 `needs_u` / `needs_b`：真答案到底需要哪些 `(op, i)` / `(op, i, j)` 表格
cell，取自生成时对标准表达式求值的日志（`instance.py:454, 264-268`）。

### Held pairs（`instance.py:159-210`）

`held_pairs` 是一组（外层, 内层）结构算子对，从 demo/iid/depth 分布里排除、在 `comp` 里被要求
出现。只有文法能实现的算子对才是候选；目标占比用"最大余数法"精确控制在约 30–33%，避免朴素
笛卡尔抽样导致的漂移。

### Tail（`instance.py:339-348`）

`is_tail(t)` = "这次 run 从没买过这一项需要的某个表条目吗？"——它是对某次 **run 的实时**查询
日志算的，不是生成时固定的，因为"没见过"只有在 agent 探测之后才知道。

## 7. π 的测量

π 刻画的是**难度落在哪**——skeleton 占总难度的份额。`measure.py:1-18`：

```
a_skel = accuracy(真 skeleton + identity 表)   # 只缺表
a_tab  = accuracy(平凡 skeleton + 真表)         # 只缺 skeleton
L_table = 1 - a_skel                            # 不知道表带来的损失
L_skel  = 1 - a_tab                             # 不知道 skeleton 带来的损失
π = L_skel / (L_skel + L_table)                 # SKELETON 的份额
```

- **π → 1**：几乎所有损失都来自不知道 *skeleton*（表很容易/近乎平凡）→ "代码/推理"型实例。
- **π → 0**：几乎所有损失都来自不知道 *表* → "记忆/训练"型实例。

`measure_pi(inst, sample=1500)`（`measure.py:75-90`）在 `inst.test[:1500]`（满规模下全是 `iid`）
上测，返回 `pi_components`：`{full, a_skel, a_tab, L_table, L_skel, pi, n}`。

**只有 π 的测量**用宽松的**digit-match**（数字位一致的比例，`measure.py:28-35`）；agent 的
arm 和参照 oracle 都用**精确匹配**（`measure.py:44-46`）。整个代码库里**没有任何 p 值 / 显著性
检验**——π 是一个点估计，连同它的原始分量一起报出来以便复算。

## 8. 确定性与 fingerprint

`fingerprint(inst)`（`frozen.py:13-23`）是对四个部分按序做的 SHA-256：demos（`expr, answer`）、
test 项（`split, expr_src, answer_src`，按生成顺序）、held pairs（**排序后**，使算子对的生成
顺序不影响哈希）、val 项（`expr_src, answer_src`）。它钉死了实例的完整身份。

`load_instance(id, verify=True)`（`frozen.py:60-85`）用 `generate(seed, PRESETS[preset])` 重生成，
并断言 fingerprint 与固化 manifest 一致，漂移就报错。因为生成是单 RNG 确定性的（§6），这保证能
**精确**复现固化数据——这也是那 15 个 benchmark 实例能"按 id 复用"而**从不存**其（庞大的）展开
数据的原因。

---

*配套：**数据验证** —— arm 与参照 oracle 是如何对一个实例做测量的。*
