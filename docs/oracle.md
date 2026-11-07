# 当前动作空间的 Oracle 测试

Oracle 从已完成采集的 `evaluation_start.pt` 恢复相同时序模型、优化器、待反馈标签、未完成批次和随机数状态，单独进行连续测试。不需要训练选择器，也不重新测试 Native/Frozen。

在项目根目录运行：

```bash
python scripts/evaluate_oracle.py \
  --data runs/collect-onenet-ettm1-s96-seed0 \
  --output runs/oracle-onenet-ettm1-s96-seed0
```

等价模块入口：`PYTHONPATH=src python -m onlinetsf.decision oracle --data ... --output ...`。

默认候选集使用采集 manifest 中保存的动作表，不依赖后来修改的配置。可通过 `--actions configs/actions/onenet_ettm1.yaml` 显式指定另一个合法动作表。`--steps 256` 可限制测试预测起点数量，用于先检查运行；省略时测试原完整区间。输出目录必须尚不存在。

## 选择和执行

每个决策位置，从同一个当前 Oracle 轨迹状态及随机数状态分别运行候选动作 a 随后的 H_a 个预测起点，使用这些预测的完整 horizon、所有目标通道上的 MAE 选择最低者。比较的是离线标准化尺度上的 MAE，不是相对 Native 的收益。精确并列时优先 Native，再按动作表顺序。

选择后，在真实 Oracle 轨迹上执行所选动作 H_a 步，保留更新后的模型、优化器和内部状态，在到期位置继续分支选择。普通动作边界不强行清空批次；动作切换沿用原控制器的尾批规则。候选超过测试末尾时，只评价并执行剩余预测起点；只有到达整段测试最后一次预测时才完成已到达样本的尾批。

未来真实标签用于选择和事后评分，分支及真实执行中的梯度更新仍通过原有延迟反馈流程。候选搜索恢复随机数状态，未选分支不改变真实轨迹。脚本会检查每段实际执行的 MAE/MSE 与获选分支评分一致。

这是逐段贪心、使用未来信息的事后参考；不同 H 对应不同局部窗口，最终比较使用同一整段测试区间的误差。它不保证整条轨迹全局最优。

## 输出

- `manifest.json`：采集来源、动作数、测试区间和运行状态。
- `actions.yaml`：本次实际使用的候选集。
- `comparison.csv`：仅 Oracle 一条汇总记录，可与已有普通测试的 `comparison.csv` 对照；不修改已有结果。
- `oracle_greedy/predictions.npy`、`targets.npy`：每个预测起点的完整预测和真实值，原始数值单位。
- `oracle_greedy/forecasts.csv`：每步动作、决策编号、标准化和原始单位的误差。
- `oracle_greedy/decisions.jsonl`：各决策时刻的状态特征、完整动作配置、候选评分、所选 MAE/MSE、实际执行步数和搜索耗时。
- `oracle_greedy/action_outcomes.csv`：每次决策中每个候选动作的 H、评分步数、MAE/MSE 和是否获选。
- `oracle_greedy/summary.json`：整段预测误差、实际更新成本、候选评价次数和候选搜索耗时。

`online_seconds` 包含候选搜索和结果写入，`candidate_search_seconds` 单独记录搜索耗时，`update_seconds` 只统计真实轨迹的更新。Oracle 需要在每个决策位置运行所有候选，因此通常比普通策略测试耗时更长。

GPU 使用采集快照保存的逻辑设备，按已有方式设置 `CUDA_VISIBLE_DEVICES`。脚本不新增设备迁移参数。

## 测试

```bash
python -m pytest tests/test_oracle.py tests/test_selectors.py -q
```

覆盖 OneNet、FSNet、DSOF 的分支隔离、随机数恢复、最小 MAE 选择、不同 H 的决策时刻、末尾截断和尾批、Native 并列优先、指定动作 YAML、CLI 和保存预测的指标复算。
