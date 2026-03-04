# ECoFLaP 剪枝 Calibration 配置说明

## 文件名标注（旧/新）

| 用途           | 脚本 | 配置（eval） |
|----------------|------|------------------|
| **原 / CC3M calibration** | `ecoflap_zeroth_cc3m_calibration.py` | `cc_prefix_derivative_compute_cc3m.yaml` |
| **现 / OK-VQA calibration** | `ecoflap_zeroth.py`（当前入口） | `cc_prefix_derivative_compute_okvqa.yaml` |

## 改动的关键位置（OK-VQA 版）

- **ecoflap_zeroth.py**
  - `job_id` 前缀：`cc3m-` → `okvqa-`（与剪枝、评测两处一致）
  - `--cfg-path`：`cc_prefix_derivative_compute.yaml` → `cc_prefix_derivative_compute_okvqa.yaml`
  - 原 CC3M 的剪枝整段已注释保留，标有 `【原 CC3M calibration 的剪枝代码（已注释）】`

- **配置**
  - `cc_prefix_derivative_compute_okvqa.yaml`：使用数据集 `prefix_okvqa_calibration`，数据路径在 `lavis/configs/datasets/okvqa/calibration.yaml` 中修改。

## 用法

- 用 **OK-VQA** 做 calibration（当前默认）:
  ```bash
  python scripts/blip2/ecoflap_zeroth.py <GPU> <port>
  ```
  对应配置：`cc_prefix_derivative_compute_okvqa.yaml`

- 用 **CC3M** 做 calibration（恢复原逻辑）:
  ```bash
  python scripts/blip2/ecoflap_zeroth_cc3m_calibration.py <GPU> <port>
  ```
  对应配置：`cc_prefix_derivative_compute_cc3m.yaml`

## 原始未改名的文件

- `cc_prefix_derivative_compute.yaml`：原 CC3M 配置，顶部已注明带标注副本为 `cc_prefix_derivative_compute_cc3m.yaml`
- `okvqa_prefix_derivative_compute.yaml`：OK-VQA 配置，顶部已注明带标注副本为 `cc_prefix_derivative_compute_okvqa.yaml`
