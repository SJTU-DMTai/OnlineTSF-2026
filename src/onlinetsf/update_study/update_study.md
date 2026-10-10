# FSNet / OneNet 在线更新实验

入口为 `python -m onlinetsf.update_study.run`，代码位于 `src/onlinetsf/update_study/`。默认读取 `runs/real-drift-electricity-5pct-20261010/`，运行 ETTh2、ETTm1、Electricity，预测长度 H=1、24，seed=0。新回放和分析代码不依赖旧 A/B/C 实验框架。

## 1. 两组实验

**when / how：** 从同一离线初始化分别重放全流，比较 frozen、no_grad、每步更新、每4步更新、每16步更新。三种梯度更新频率各测试学习率倍率 0.3、1、3，共11条轨迹。`native` 是每步、倍率1的基准，不重复运行。也在稀疏节点做同状态分支，区分长期轨迹差异与当前状态下的策略收益。

**what：** 在同一个探针状态下，枚举每种方法三个参数组的8种开关组合。全开复用 native，全关复用 no_grad，另外6种为单组和双组更新；每种组合每步执行，使用基准学习率，不与全部频率和学习率做笛卡尔积。默认联合实验每个探针共17个候选。

| 方法 | 参数组 | 具体范围 |
| --- | --- | --- |
| OneNet | `time` | Time-TCN专家全部参数 |
| OneNet | `variable` | 跨变量TCN专家全部参数 |
| OneNet | `decision` | 短期决策MLP全部参数 |
| FSNet | `base` | 输入投影、适配卷积的原始卷积权重和偏置、残差投影 |
| FSNet | `adapter` | 梯度控制器、weight_scale、bias_scale、feature_scale |
| FSNet | `head` | 最终多步预测线性头 |

保留未选参数的 `grad=None`，阻止 Adam/AdamW 动量和权重衰减继续改变它们。FSNet参数分组互斥且覆盖所有参数。冻结参数组不等于冻结它的非梯度内部状态。

## 2. no_grad 的准确含义

这里的 no_grad 表示**不执行神经网络参数的梯度下降**，并不表示把所有计算包在 `torch.no_grad()` 中。

| 行为 | native | 每4/16步未训练时 | no_grad | frozen |
| --- | --- | --- | --- | --- |
| 选定神经网络参数训练 | 是 | 否 | 否 | 否 |
| FSNet梯度计算、快慢梯度EMA | 是 | 是 | 是 | 否 |
| FSNet适配状态与关联记忆演进 | 是 | 是 | 是 | 否 |
| OneNet公式EGD | 是 | 是 | 是 | 否 |
| OneNet用固定/当前MLP前向刷新short_bias | 是 | 是 | 是 | 否 |

FSNet的非梯度状态依赖真实梯度，因此 no_grad 仍执行反向传播来获得梯度，再更新记忆，最后丢弃参数梯度；其计算成本可能仍较高。OneNet的 no_grad 不执行神经网络反向传播，保留原生MLP前向模式（包括原生训练模式下的dropout）。

OneNet长期权重已按用户确认改为[论文式(2)](https://papers.neurips.cc/paper_files/paper/2023/file/dd6a47bc0aad6f34aa5e77706d90cdc4-Paper-Conference.pdf)的EGD。论文区分长期EGD与短期MLP；原项目的Adam长期权重实现已经移除。每个目标变量使用预测时缓存的两位专家输出，标签完整到达后计算各自的预测步平均平方误差：

`log(w_time / w_variable) += weight_learning_rate * (variable_mse - time_mse)`

这是归一化乘法EGD的稳定对数形式，不对网络求导。保留项目已有的logit短期修正与OneNet-TCN结构；本实验不是宣称完整复现论文所有结构与基准。`n_inner` 沿用profile，每轮均推进原生更新过程。

学习率倍率**只调整神经网络优化器**：FSNet优化器，OneNet专家和决策MLP优化器。EGD速率在所有候选中固定，避免把梯度更新频率/LR与公式权重更新混在一起。各组件的基准学习率来自profiles，可用单独的profiles文件设置。倍率只在离线训练完成后施加；所有候选共享同一离线结果。旧实验的OneNet结果使用不同权重更新算法，不能直接当作本轮native结果。

旧 `scripts/drift_experiment_runtime.py` 的FSNet/OneNet跳过梯度路径也已修正为保留上述非梯度更新。

## 3. 时间与输入约束

- 默认上下文L=96、stride=1；只选择manifest列出的原始数值列，同时预测全部这些变量。日期和全部标注列不进入模型。
- 默认离线前缀为 `max(7天标注初始化行数, ceil(原始总行数×20%))`；它仅是模型离线训练前缀，不声明该区间没有漂移。`--prefix-rows` 可以增大前缀，但不能低于该下限。20%按原始行数计算，不按滑动窗口数计算。
- ETTh2：总17,420行，7天168行（0.9644%），离线前缀改为3,484行。ETTm1：总69,680行，7天672行（0.9644%），改为13,936行。Electricity：总26,304行，7天168行（0.6387%），改为5,261行（20%向上取整）。
- 归一化使用该离线前缀的固定逐变量z-score，不使用检测器的sigmoid变换。所有结果在z-score尺度报告。
- 训练窗口完整目标全部位于前缀内；首个在线预测的第一个目标是 `prefix_rows` 原始行，避免与离线标签重叠。
- 窗口编号i的预测起点原始行是 `i+L-1`，目标为 `[i+L, i+L+H)`，完整反馈在窗口编号 `i+H` 到达。先接收当前成熟反馈、更新，再发出当前预测。
- 待反馈队列只保存已经发出的预测及其起点，不在快照中保存未来标签。误差只用原预测计算；末尾flush仅评分，不训练或推进适配状态。
- 目标级阶段指标按目标实际原始行归属；更新日志阶段按反馈到达原始行归属。二者独立记录。

## 4. 有限分支

默认从原生每步轨迹选最多12个探针，最多抽8个事件；事件按时间分层抽样。候选包括共识事件前16行、事件首行、事件结束后16行，以及足够长平稳候选区间内的随机节点。四类轮流选取，全局最小间隔默认64行，不会每个事件、每个节点全部分叉。平稳探针要求上下文及整个评价块的目标均在平稳候选区内；事件附近分支允许跨阶段。

每个探针在接收该时刻成熟反馈之前复制完整状态：模型、优化器、FSNet记忆、OneNet待评分专家预测、历史误差和待反馈队列。所有候选恢复同一CPU/CUDA随机数起点，分支结束后恢复主轨迹随机数状态。探针不改变主轨迹。

每个候选持续执行128个预测起点，报告前8、32、128个预测窗口的平均MAE/MSE。H=24时这些是滚动24步预测窗口，不能解释为8/32/128个不重叠目标。分支之前发出的预测仍按时消费并可能用于训练，但不计入分支预测收益。短lookahead成本不单独归因，日志成本对应完整分支。

`pre_event` 等类别使用事后伪标签，不是线上触发器。共识事件边界也不是已验证的真实概念漂移起止点。默认阈值、分支长度和采样量是可调整的实验起点。

## 5. 执行

尚未安装为可编辑包时，在项目根目录设置：

```powershell
$env:PYTHONPATH='src'
```

先查看实际实验量（会加载并校验输入，但不建模型或训练）：

```powershell
.venv/Scripts/python.exe -m onlinetsf.update_study.run --plan-only --output runs/update-study-plan
```

先做少量分支探索：

```powershell
.venv/Scripts/python.exe -m onlinetsf.update_study.run --datasets ETTh2 --scope branches --max-probes 4 --device cpu --output runs/update-study-etth2-branches
```

完整运行全流when/how和分支when/how/what，可在有CUDA的机器上执行：

```powershell
python -m onlinetsf.update_study.run --scope both --suite both --seeds 0 1 2 --device cuda --output runs/update-study-main
```

只做模块选择：`--suite what --scope branches`。只做全流频率/LR：`--suite when --scope full`。`--max-steps` 可限制在线预测起点用于小样本验证；短流上可能没有满足完整分支长度的节点，计划文件会如实记录0个探针。输出目录必须不存在，避免覆盖实验。

如需逐值审计，加 `--save-predictions`，以每256个预测为一块保存实际预测和真实目标张量；默认保留逐窗口误差及逐阶段/预测步聚合，避免Electricity多变量多步输出造成很大存储开销。

### 远程Linux四GPU并行

在服务器项目根目录、已安装CUDA版PyTorch的环境中执行。脚本使用四个独立的nohup进程：GPU0运行FSNet/H=1，GPU1运行FSNet/H=24，GPU2运行OneNet/H=1，GPU3运行OneNet/H=24。每个进程依次运行三个数据集和seed=0/1/2，覆盖全部组合而不重复离线初始化。各进程中的 `--device cuda` 指向其唯一可见GPU。

```bash
python -m pip install -e .
bash scripts/run_update_study_4gpu.sh runs/update-study-main-4gpu
```

脚本分别设置 `CUDA_VISIBLE_DEVICES=0/1/2/3`、单线程CPU环境变量，保存nohup日志和PID。如需指定Python解释器：

```bash
PYTHON_BIN=/path/to/venv/bin/python bash scripts/run_update_study_4gpu.sh runs/update-study-main-4gpu
```

根输出目录必须不存在。启动后查看日志：

```bash
tail -f runs/update-study-main-4gpu/logs/gpu0.log
```

四个进程全部完成后合并：

```bash
PYTHONPATH=src python -m onlinetsf.update_study.analyze runs/update-study-main-4gpu
```

汇总器按任务目录读取，不重复累加各GPU根目录的已有汇总。远程运行前可执行本页针对性测试；新实验本身不需要CapyMOA。

## 6. 产物与规律分析

| 产物 | 用途 |
| --- | --- |
| 根manifest.json | 输入SHA256、参数、profile、版本、前缀、探针列表、计算量及完成状态 |
| 各任务initial.pt | 同一离线初始化的完整方法对象、优化器状态和随机数状态 |
| policies.json | 所有候选参数组、频率、学习率倍率 |
| 各轨迹forecasts.csv | 预测起点、目标区间、反馈时刻、原预测MAE/MSE，OneNet另含两专家MAE |
| updates.csv | 更新发生时刻和阶段、参数组、累计实际优化器步数、EGD次数及反馈计算时间 |
| phase_metrics.csv | 按真实目标行上的伪标签阶段和预测步聚合误差及值数量 |
| probe_states.csv | 在线可得输入统计、成熟残差、当前反馈误差、更新历史及方法内部状态；事后事件信息另列 |
| probe_outcomes.csv | 每个同状态候选在各lookahead上的未来预测误差 |
| 根summary.csv | 全流预测误差与实际更新次数、EGD次数和耗时 |
| paired_benefits.csv | 同探针候选相对native/frozen的配对收益，正值表示候选更好 |
| best_actions.csv | 有限候选集内的事后较优动作和native regret |
| pattern_summary.csv | 按探针类别、方法、H、seed及策略汇总收益 |
| relationships.csv | 在线状态特征与动作收益的探索性Pearson相关，至少4个有效探针且特征/收益有变化才输出 |
| drift_relationships.csv | 事件长度、峰值票数等事后标注特征与收益的关系，禁止作为在线输入直接使用 |

在线状态统计包含32/128行均值、尺度、变化幅度、趋势、自相关，最多16个固定变量的跨变量相关，历史残差均值/偏差、距上次训练的间隔、近期训练比例，以及OneNet权重/分歧/专家误差或FSNet梯度记忆状态。读取这些特征不执行额外模型前向，不推进适配状态。

重新生成分析：

```powershell
python -m onlinetsf.update_study.analyze runs/update-study-main
```

事后较优不等于线上可获得的最优策略；相关不是因果关系。多步窗口和同一事件的探针相互依赖，不作为独立重复实验。首先看同一状态的收益方向是否随输入/反馈变化，再用独立时间段、多seed和不同历史策略验证；控制器训练时不能使用事后事件长度或未来收益作为输入。仅原始观测上的共识报警不能单独证明条件预测关系改变。

## 7. 验证

在远程服务器执行新增的独立测试：

```bash
python -m pytest tests/test_update_study.py -q
```

如果同步了项目已有测试，还可以运行：

```powershell
.venv/Scripts/python.exe -m pytest tests/test_update_study.py tests/test_fsnet.py tests/test_online.py -q
```

测试覆盖EGD公式、no_grad参数与优化器冻结、FSNet记忆演进、全部单组/双组参数掩码（含已有Adam动量）、frozen状态冻结、反馈延迟、未来标签隔离、末尾只评分、分支深复制、主轨迹随机数恢复、标注列隔离、前缀归一化及20%下限、选点边界、阶段归属、四GPU汇总去重和分析重跑一致性。完整现有测试涉及CapyMOA时，需使用可用的JDK并清空有问题的CLASSPATH；本实验本身不导入漂移检测器或启动JVM。

新增测试文件已单独从.gitignore的tests忽略规则中放行，便于同步到服务器。按用户要求，最终修改只做本地静态检查，CUDA路径留待服务器验证。
