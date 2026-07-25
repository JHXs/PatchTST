# PatchTST

基于 PyTorch、fastai 和 tsai 的时间序列预测实验项目，包含 PatchTST、ST-PatchTST、
CT-PatchTST 和 Informer 模型。

## 环境说明

- 推荐使用 64 位 Python 3.12。
- GPU 训练使用 AMD ROCm。PyTorch 在 ROCm 环境下仍然使用
  `torch.cuda` 接口，因此项目代码不需要改成 `torch.rocm`。
- `requirements.txt` 是从 Linux Conda 环境导出的快照，包含
  `file:///home/...` 路径和 Linux 专用包，不能用于跨平台安装。
- Windows 和 Linux 均使用 uv、`pyproject.toml` 和 `uv.lock` 管理环境。
- `pyproject.toml` 会根据操作系统选择 PyTorch 软件源：Windows 使用 AMD
  ROCm 7.14 `gfx1102` 软件包，Linux 使用项目当前的 ROCm 7.2 软件包。

## Windows 11 环境搭建

以下流程面向 AMD Radeon RX 7600。RX 7600 的 ROCm 架构标识是 `gfx1102`。

### 1. 准备系统环境

1. 将 Windows 11 更新到 AMD 当前验证的版本，推荐 Windows 11 25H2。
2. 安装适用于 RX 7600 的最新版 AMD Software: Adrenalin Edition 驱动。
3. 根据 AMD 文档完成 ROCm Core SDK 的 Windows 前置条件。
4. 安装 64 位 Git，并确认它已经加入 `PATH`。Python 3.12 由 uv 管理。

参考文档：

- [ROCm 7.14 发布与系统支持](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html)
- [AMD ROCm 安装说明](https://rocm.docs.amd.com/en/latest/install/rocm.html)
- [AMD ROCm PyTorch 安装说明](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)

### 2. 安装 uv

在 PowerShell 中执行：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

重新打开 PowerShell，然后检查安装：

```powershell
uv --version
```

### 3. 同步项目环境

在 PowerShell 中进入项目目录：

```powershell
cd C:\path\to\PatchTST
uv python install 3.12
uv sync --python 3.12
```

uv 会创建 `.venv`，并根据 Windows 平台自动从 AMD ROCm 7.14 软件源安装
RX 7600 对应的 `gfx1102` PyTorch。不要再用 `pip` 单独安装或升级
`torch`、`torchvision`，否则可能破坏锁定环境。

### 4. 验证 GPU

```powershell
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('ROCm/HIP:', torch.version.hip); print('GPU available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

正常情况下应看到：

- `ROCm/HIP` 显示 ROCm/HIP 版本；
- `GPU available: True`；
- GPU 名称包含 `AMD Radeon RX 7600`。

如果输出为 `False`，先不要运行训练，检查显卡驱动、ROCm 安装和 PyTorch
是否确实来自 AMD 的 `gfx1102` 软件源。

## Linux 环境搭建

以下流程面向 x86-64 Linux 和 AMD Radeon GPU。安装 Python 环境前，请先按照
[AMD ROCm Linux 安装文档](https://rocm.docs.amd.com/en/latest/install/rocm.html)
安装与显卡、Linux 发行版匹配的驱动和 ROCm 运行环境。

### 方案 A：使用 uv（项目默认方式）

项目的 `uv.lock` 和 `pyproject.toml` 已配置 Linux ROCm 7.2 软件源，适合复现
项目当前环境。

安装 uv：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

进入项目并同步 Python 3.12 环境：

```bash
cd /path/to/PatchTST
uv python install 3.12
uv sync --python 3.12
```

验证 PyTorch 和 GPU：

```bash
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('ROCm/HIP:', torch.version.hip); print('GPU available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 方案 B：RX 7600 使用 ROCm 7.14

RX 7600 的架构是 `gfx1102`。如果项目锁定的 ROCm 7.2 环境不能识别显卡，
可以使用 uv 创建独立环境，并安装 AMD ROCm 7.14 的 `gfx1102` 构建。这个
方案独立于项目当前的 ROCm 7.2 锁定环境：

```bash
uv venv --python 3.12
uv pip install --index https://repo.amd.com/rocm/whl-multi-arch/ \
  "torch[device-gfx1102]==2.12.0+rocm7.14.0" \
  "torchvision[device-gfx1102]==0.27.0+rocm7.14.0" \
  "torchaudio==2.11.0+rocm7.14.0"
uv pip install \
  "tsai>=0.4.1" \
  "fastai>=2.8.7" \
  "numpy>=1.26" \
  "pandas>=2.2" \
  "scipy>=1.17" \
  "scikit-learn" \
  "fastcore>=1.13.2,<2" \
  "jupyter-core>=5.9.1"
```

验证 GPU：

```bash
uv run --no-sync python -c "import torch; print('PyTorch:', torch.__version__); print('ROCm/HIP:', torch.version.hip); print('GPU available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

这里需要使用 `uv run --no-sync`，避免 `uv run` 自动把独立实验环境重新同步为
项目锁定的 ROCm 7.2 环境。

## 准备数据

原始站点 CSV 文件需要位于以下目录之一：

- `tsai/data/stations_data/`
- `tsai/data/stations_data_Guangzhou/`

多站点实验使用：

```bash
python data_preparation.py
```

单站点 PatchTST 基线使用：

```bash
python data_preparation_single.py
```

两个脚本都会写入 `tsai/data/X.npz`、`y.npz`、`splits.pkl` 和预处理对象，
因此一次实验只运行其中一个，避免后运行的脚本覆盖前一个脚本生成的数据。

使用 uv 时，在命令前加 `uv run`，例如：

```bash
uv run python data_preparation.py
```

## 训练与验证

在 `main.py` 中通过 `train_model` 选择模型：

- `pa` 或 `patchtst`：PatchTST；
- `st`：ST-PatchTST；
- `ct`：CT-PatchTST；
- `informer`：Informer。

运行训练和评估：

```bash
python main.py
```

Linux 使用 uv 时：

```bash
uv run python main.py
```

提交修改前可在 Linux 中执行快速语法检查：

```bash
python -m py_compile *.py
```

Windows PowerShell 中可以执行：

```powershell
Get-ChildItem -Filter *.py | ForEach-Object { python -m py_compile $_.FullName }
```
