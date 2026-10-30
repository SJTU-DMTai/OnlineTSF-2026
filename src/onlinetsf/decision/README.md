# 在线更新动作执行层

当前动作仅包含四个维度：

| 维度 | Python 字段 | 含义 |
| --- | --- | --- |
| `M` | `units` | 启用的参数或内部适配状态单元；空集表示冻结 |
| `η` | `learning_rate_scales` | 各选中单元相对于方法默认学习率或适配步长的正倍率 |
| `F` | `periods` | 各选中单元的正整数更新周期 |
| `H` | `duration` | 配置在 `[t, t + H)` 生效，在 `t + H` 重新决策 |

本模块只执行原生在线更新，不支持重训练。初始离线训练仍由原有实验流程完成。策略通过 `policy(DecisionState) -> UpdateAction` 注入；策略学习、专家路由和专家池管理不属于本模块。

## 模块与执行顺序

| 文件 | 职责 |
| --- | --- |
| `actions.py` | 四维动作、数值校验、JSON 序列化 |
| `capabilities.py` | 可更新单元声明、方法相关合法性、独立周期计数 |
| `optimizers.py` | 拆分参数组以独立设置学习率，保留原优化器及其统计量 |
| `adapters.py` | OneNet、FSNet、DSOF 的原生机会识别、反馈记录和单元执行 |
| `runtime.py` | 整理当前反馈、调用策略、配置到期与切换、更新成本统计 |

每个数据时刻，`OnlineExecutor` 通知当前观测，收集当前已到达的全部反馈；控制器先整理必要记录，在首次执行或配置到期时决策，然后按原生顺序更新并预测。运行结束的 `flush()` 只整理标签和评分，不训练、不重新决策。

不再保存用于重训练的完整样本历史。OneNet 保留待反馈的专家预测；DSOF 保留未完成窗口的部分标签、预测诊断，以及底层方法容量受限的原生回放。冻结时也维护这些必要记录，以便恢复在线更新。

`DecisionState` 提供当前索引、输入窗口、当前到达的反馈、方法能力、上次动作及各单元的机会计数副本。需要更多诊断时，策略对象可持有方法引用并读取其原生状态。

## 单元与周期语义

| 方法 | 单元 | 合法更新机会 | 独立学习率 |
| --- | --- | --- | --- |
| OneNet | `time`, `variable`, `decision` | 完整反馈 | 有 |
| OneNet | `long` | 完整反馈下的长期组合更新 | 有，保持指数梯度原生步长含义 |
| OneNet | `short` | 完整反馈下的短期修正刷新 | 无 |
| FSNet | `base`, `adapter`, `head` | 完整反馈 | 有 |
| FSNet | `gradient_memory` | 完整反馈下的原生反向传播过程 | 无 |
| FSNet | `adaptation_memory` | 原生推进适配状态的前向：预测及内部训练的后续前向 | 无 |
| DSOF | `teacher`, `student` | 样本足够且原生 `replay_frequency` 触发 | 有 |
| DSOF | `online` | 新到达的首步标签及当前输入窗口 | 有，使用独立在线优化器 |

周期在单元的首个合法机会执行，随后每隔 `p` 个合法机会执行一次。周期为 1 保留原生机制及触发条件；底层方法的内部训练次数保持原有配置。

相邻动作中仍启用且周期不变的单元延续计数，学习率倍率或保持时长变化不重置相位。停用后重新启用或改变周期时，从首个合法机会重新计数。更新周期与配置保持时长相互独立。

FSNet 可以仅启用 `gradient_memory`：执行所需反向传播并更新梯度记忆，但不执行优化器步进。只启用 `adaptation_memory` 时不反向传播，检索、融合和写回保留原生触发条件。

DSOF 的 `student` 和 `online` 更新同一套学生参数，但使用不同监督和优化器，可以独立选择、设置学习率和周期。原生回放属于在线更新机制。

## Python 接口

```python
from onlinetsf.decision import UpdateAction, UpdateController
from onlinetsf.online import OnlineExecutor

# method 是已构造、可选已完成初始离线训练的 OneNetMethod。
action = UpdateAction(
    units={"time"},
    learning_rate_scales={"time": 0.5},
    periods={"time": 4},
    duration=128,
)
controlled = UpdateController(method, action)
executor = OnlineExecutor(controlled, feedback_delay=method.model.horizon)
run = executor.run_dataset(dataset)
```

实际在线学习率 = 方法的在线基准学习率 × `learning_rate_scales`。例如，时间专家的在线基准学习率为 `0.001`，倍率为 `0.5` 时，实际学习率为 `0.0005`。倍率为 1 或未指定时，使用在线基准学习率。每次切换配置均相对于同一基准值计算，不在上次实际学习率上累乘。

所有方法的离线训练使用独立优化器和学习率，不共享在线优化器的动量或二阶统计量，也不受在线动作倍率影响。`method` 中的基准学习率仍是绝对值；只有动作中的 `learning_rate_scales` 是倍率。

方法默认学习率通过现有 `method` 配置设置：

| 方法 | 更新单元 | 默认值的配置字段 |
| --- | --- | --- |
| OneNet | `time`, `variable` | `learning_rate` |
| OneNet | `decision` | `decision_learning_rate` |
| OneNet | `long` | `weight_learning_rate` |
| FSNet | `base`, `adapter`, `head` | `learning_rate` |
| DSOF | `teacher` | `learning_rate` |
| DSOF | `student` | `student_learning_rate` |
| DSOF | `online` | `online_learning_rate` |

直接使用 Python 接口时，默认值取自构造控制器时各优化器参数组的学习率，以及 OneNet 的长期组合步长。选中但未指定周期的单元使用周期 1。没有步长的状态单元不得设置倍率；倍率和周期只能指向选中的单元。

离线训练使用 `method.offline_learning_rate`；OneNet 另有 `offline_decision_learning_rate`、`offline_weight_learning_rate`，DSOF 另有 `offline_student_learning_rate`。Under-Cali 的 `uncertainty_lr` 用于离线不确定性训练，`uncertainty_adapt_lr` 用于在线更新，两者使用独立优化器。

DSOF 当前统一采用官方 DLinear + MLP、ETTh2 的原始设置，不按数据集切换：

| 阶段与对象 | 基准学习率 | 优化器 |
| --- | --- | --- |
| 离线 teacher | `0.005771621537901884` | Adam，AMSGrad=True，weight_decay=0 |
| 离线 student | `0.001026953060677239` | AdamW，AMSGrad=True，weight_decay=0.01 |
| 在线 ER teacher | `0.00028889418723619334` | AdamW，AMSGrad=True，weight_decay=0.01 |
| 在线 ER student | `0.00015686974622312612` | AdamW，AMSGrad=True，weight_decay=0.01 |
| 在线 TD student | `0.00036288062287989935` | AdamW，AMSGrad=True，weight_decay=0.01 |

数值与优化器选项来源：[官方 DLinear + MLP / ETTh2 配置](https://github.com/yyalau/iclr2025_dsof/blob/11cebd980a1f77ef0e1db9ac313dbfe6d37984e9/config/optimizer/w_student/DLinear_MLP/ETTh2.yaml)。

例如 `learning_rate_scales={"student": 2.0, "online": 0.5}` 将 ER student 学习率设置为其基准的 2 倍、TD student 设置为其基准的 0.5 倍，离线两套优化器保持原值。`reset_online_state()` 清理在线状态，不清除离线优化器的统计量。原有 checkpoint 不提供兼容转换，修改模型或优化器配置后应重新生成初始 checkpoint。

完整原生动作为 `controlled.capabilities.native_action(duration=128)`，冻结为 `UpdateAction(duration=128)`。OneNet 的 no_grad 范围是 `{"long", "short"}`，FSNet 是 `{"adaptation_memory"}`，DSOF 与冻结相同。

固定动作在每次到期后重新选择同一配置，连续启用单元的计数按上述规则延续。也可通过可调用策略动态选择：

```python
class ExamplePolicy:
    def __call__(self, state):
        if state.feedback:
            return state.capabilities.native_action(duration=32)
        return UpdateAction(duration=1)

controlled = UpdateController(method, ExamplePolicy())
```

索引采用 `OnlineExecutor` 的数据时刻，要求连续递增；标准滑窗数据使用 `stride: 1`。无反馈的时刻也在到期时重新决策。`duration` 不按标签事件数或优化器步数计量。

## YAML 入口

在现有实验 YAML 中添加：

```yaml
decision:
  action:
    units: [time, long, short]
    learning_rate_scales: {time: 0.5, long: 1.0}
    periods: {time: 4, long: 1, short: 1}
    duration: 128
```

运行 `python -m onlinetsf --config ...`。OneNet、FSNet 的完整反馈延迟不得小于预测长度；DSOF 使用 `[1, ..., horizon]`。动作与优化器步数、决策和更新耗时沿用现有实验输出。

旧的大动作空间中的 `operation`、`retrain`、`history_capacity` 等字段及 `RetrainConfig` 已删除，不提供兼容转换。旧的 17 动作实验仍独立保存在 `onlinetsf.decision_previous`。

## 验证

运行 `python -m pytest tests/test_update_actions.py -q`。测试覆盖 32/32/8 个在线范围、原生轨迹一致性、各模块独立学习率和周期、冻结与恢复、梯度记忆单独执行、反馈因果性、有界回放与部分标签、序列化续跑、标准实验入口和废弃配置拒绝。
