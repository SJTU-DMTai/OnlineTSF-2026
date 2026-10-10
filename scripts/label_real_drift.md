# 真实时序数据的五检测器共识标注

`label_real_drift.py` 标注本项目的 ETTh2、ETTm1、Electricity CSV。仅输入数值观测，日期用于时间对齐；不加载预测模型，不使用预测或预测残差。ABCD 使用算法自身的 PCA 重构检测机制。

## 执行

依赖为项目现有的 NumPy、PyYAML、River 和可选 CapyMOA。SEED 需要兼容的 JDK；本工作区可使用 JDK 21：

合并间隔和过渡区必须分别通过 `--merge-gap-samples` 和 `--transition-samples` 指定非负整数，单位是采样点，不按采样频率换算，也不提供未经验证的默认值。以下命令中的 8 和 32 仅为用法示例，不是经过实验选择的推荐值。

```powershell
$env:CLASSPATH=''
.venv/Scripts/python.exe scripts/label_real_drift.py --merge-gap-samples 8 --transition-samples 32 --java-home 'C:/Program Files/Java/jdk-21'
```

本工作区原有 `CLASSPATH` 包含带引号的旧路径，JPype 会报 `InvalidPathException`；上面的清空只作用于当前 PowerShell 会话。

默认处理 `data/ETTh2.csv`、`data/ETTm1.csv`、`data/electricity.csv` 的全部变量和全部行，输出到 `runs/real-drift-labels-<UTC时间>/`。指定其他输入/输出目录：

```powershell
python scripts/label_real_drift.py --data-dir data --output runs/real-drift-annotations --merge-gap-samples 8 --transition-samples 32 --workers 4 --java-home 'C:/Program Files/Java/jdk-21'
```

输出目录必须不存在，避免覆盖旧实验。

`--workers 4` 将标量方法的变量分到 4 个进程，各变量始终保留独立状态和固定种子，结果与串行相同。ABCD 始终接收全向量，不拆分变量；默认串行避免多进程内存开销。Electricity 的 KSWIN 通常是主要耗时项。

只处理部分数据集或验证小样本：

```powershell
python scripts/label_real_drift.py --datasets ETTh2 ETTm1 Electricity --max-rows 512 --calibration-hours 24 --merge-gap-samples 8 --transition-samples 32 --output runs/real-drift-preview --java-home 'C:/Program Files/Java/jdk-21'
```

`--max-rows` 仅用于预览，不代表完整数据集结果。ETTh2/Electricity 是每小时采样，ETTm1 是每 15 分钟采样；相同合并/过渡参数在三个数据集上均使用相同行数，对应的实际时长不同。不规则或逆序时间戳会被拒绝。初始化前缀 `--calibration-hours` 和投票容差 `--vote-window-hours` 仍按小时指定；这些现有默认值是待验证的实验设置。

## 输入预处理与检测器

先用前 168 小时的观测拟合每个变量的均值/标准差，之后固定不变；将标准化观测通过 sigmoid 映射到 [0,1]。前缀中的恒定变量使用尺度 1，并逐项记录在 manifest。这些前缀行用于初始化检测器，但不计报警统计，也不伪装成已验证的平稳区间，标签为 `calibration`。

SEED、ADWIN、KSWIN、HDDMW 分别对每个变量维护一个独立检测器；ABCD 对整行多变量向量维护一个检测器。HDDMW 复用项目已有 `source="features"` 适配器，对标准化值做相同 sigmoid 并启用双侧检测；这是数据均值检测用法，报警不能解释成模型错误率变化。

检测器参数读取 `profiles.yaml` 的 `parameters`，忽略其中预测实验专用的 `source` 和 `error_threshold`。ABCD 限定 PCA；未指定时 `max_size=2048` 以限制流式窗口大小。传给检测器的参数、库版本、输入文件 SHA256、归一化统计和随机种子均保存；其余构造参数使用对应库版本的默认值。

参数接口依据项目安装版本和 [River HDDMW 源码](https://github.com/online-ml/river/blob/main/river/drift/binary/hddm_w.py)、[CapyMOA SEED](https://capymoa.org/api/modules/capymoa.drift.detectors.SEED.html)、[CapyMOA ABCD](https://capymoa.org/api/modules/capymoa.drift.detectors.ABCD.html)。检测器参数是可复现的起点，未经真实漂移真值校准。

## 多数投票与区间定义

规则（合并和过渡长度需显式指定）：

1. 使用包含当前行的向后 6 小时时间窗。每种标量方法统计窗内曾报警的**不同变量数**，达到全部变量的 10%（向上取整）时贡献一票。ETT 需要至少 1/7，Electricity 需要至少 33/321。ABCD 在窗内发生至少一次全向量报警时贡献一票。
2. 五种方法最多各一票；至少三种投票为共识报警。方法重复报警或多个变量报警都不会增加其方法票数。不同方法可以响应不同变量，此处共识是数据集级的变化证据，不表示定位到同一变量。
3. 相邻共识报警之间的非共识行数不超过 `--merge-gap-samples G` 时合并为一个漂移事件。例如行号 10 和 14 之间有 3 行非共识报警，`G=3` 时合并，`G=2` 时不合并；`G=0` 仅合并连续报警。事件的第一行共识报警是 `drift point`；第一至最后共识报警的范围是 `drift`，包括被合并的小间隔。
4. 每个漂移区间两侧各扩展 `--transition-samples T` 行作为 `transition`，遇到初始化前缀或数据末尾时截断；`T=0` 表示不扩展。重叠过渡区取并集，`drift` 优先。剩余评价行标为 `stable`，含义是“平稳候选”，并非统计上证明平稳。前缀单独标为 `calibration`。

这些是**检测器共识伪标签**。报警通常滞后于变化；缓冲区不是识别出的渐变真实起止点。周期、季节性和自相关也可能引起报警；没有共识也不保证没有变化。区间合并与两侧扩展需要事后信息，标签只能用于分层分析，不能直接作为测试期在线策略输入。

各规则都可修改：

```powershell
python scripts/label_real_drift.py --vote-window-hours 6 --min-feature-fraction 0.1 --merge-gap-samples 8 --transition-samples 32
```

严格同一行投票用 `--vote-window-hours 0`；如果希望任意一个变量都能贡献方法票，可将 `--min-feature-fraction` 设为 `0.001`（对当前三个数据集均向上取整为一个变量）。建议对投票容差、变量比例、合并点数和过渡点数做敏感性分析，不能根据后续预测收益挑选标签规则。先前按小时生成的区间标签仍采用旧规则，需要重新生成才能反映本次修改。

## 输出

每个数据集目录包含：

| 文件 | 内容 |
| --- | --- |
| `labeled_data.csv` | 原始列原值，加逐行阶段、区间 ID、事件 ID、漂移点、共识票数、每种方法的原始报警/投票/窗内受影响变量数 |
| `drift_points.csv` | 每个合并漂移事件一个报警点，记录原始行号、区间、峰值票数和过渡区范围 |
| `intervals.csv` | 完整连续分区，含 calibration/stable/transition/drift；所有区间覆盖全部行且互不重叠 |
| `alarm_events.csv` | 每次变量级原始报警；ABCD 的变量名为 `all_features` |
| `alarm_summary.csv` | 五方法及 majority 的报警次数/率、变量报警总次数/率、投票次数/率、检测耗时 |
| `alarm_summary_by_variable.csv` | 每种标量方法每个变量的报警次数/率，以及 ABCD 全向量统计 |
| `manifest.json` | 数据来源、有效参数、规则、归一化统计、阶段行数和事件数 |

根目录另有汇总三个数据集的 `alarm_summary.csv`。所有行号从 0 开始；区间为 `[start_index, end_index_exclusive)`。

报警口径明确区分：

- `alarm_count`：该方法至少一个变量报警的**不重复时刻数**；同一行多个变量只算一次。ABCD 是全向量报警时刻数。
- `alarm_rate = alarm_count / evaluated_rows`，`alarms_per_1000_rows = 1000 × alarm_rate`。分母是校准前缀之后实际处理的行数，包含算法自身在重置/重新训练时的内部等待期。
- `variable_alarm_count`：所有独立变量报警次数之和；`variable_alarm_rate` 的分母是 `evaluated_rows × detector_instances`。
- `vote_rows`/`vote_rate`：方法经过时间容差窗和变量比例聚合后，实际贡献方法票的时刻数/比例，与原始报警率不同。
- majority 的 `alarm_count` 是共识成立的时刻数；合并后的事件数见 manifest 的 `drift_event_count` 和 `drift_points.csv`。二者不能混用。

标注列是评估元数据。后续预测实验应显式选择原始数值列，避免把标签、票数或事件 ID 当成输入特征。
