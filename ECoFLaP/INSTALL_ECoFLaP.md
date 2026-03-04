# ECoFLaP 环境安装指南

按你要跑的模块选择对应步骤，**只跑 BLIP-2/FlanT5/ViT 实验的话，完成「一、LAVIS」即可**。

---

## 一、LAVIS（BLIP-2、FlanT5、ViT）— 必装

主实验都在 `LAVIS/` 下，需要先装好 LAVIS 和其依赖。

### 1. 创建环境（推荐）

```bash
conda create -n ecoflap python=3.9 -y
conda activate ecoflap
```

### 2. 安装 PyTorch（按你机器 CUDA 版本选一个）

**说明**：原实验（LAVIS）只要求 `torch>=1.10.0`，没有写死 CUDA 版本；用你当前系统的 CUDA 即可。你系统是 **CUDA 11.8** 时，建议装 **cu118** 的 PyTorch。

通用（pip 自动选 CUDA 版本）：

```bash
pip install torch torchvision torchaudio
```

或指定版本：

```bash
# PyTorch 1.13 只有 cu116/cu117，没有 cu118。cu117 可在 CUDA 11.8 上使用（向后兼容）
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

```bash
# 若坚持要 cu118，需用 PyTorch 2.x（LAVIS 要求 torch>=1.10，2.x 兼容）
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装 LAVIS（会顺带装齐 requirements.txt 里的依赖）

```bash
cd /root/autodl-tmp/ECoFLaP/LAVIS
pip install -e .
```

**LAVIS 的 `requirements.txt` 里主要包含：**

- contexttimer, decord, einops (>=0.4.1), fairscale (==0.4.4), ftfy  
- iopath, ipython, **omegaconf**, opencv-python-headless (==4.5.5.64)  
- opendatasets, packaging, pandas, plotly, pre-commit  
- pycocoevalcap, pycocotools, python-magic, scikit-image, sentencepiece  
- spacy, streamlit, timm (==0.4.12), torch (>=1.10.0), torchvision  
- tqdm, transformers (>=4.25.0,<4.27), webdataset, wheel, POT  

`pip install -e .` 会按 `setup.py` 自动安装上述依赖，无需再单独 `pip install -r requirements.txt`。

### 4. 若报 NumPy / 缺包

- **NumPy 2.x 与 PyTorch 1.13 不兼容**：`pip install "numpy<2"`
- **ModuleNotFoundError: No module named 'datasets'**：`pip install datasets`

### 5. 验证

```bash
python -c "import lavis; print('LAVIS OK')"
```

---

## 二、CoOp（CLIP 实验）— 可选

若跑 CLIP 相关脚本（如 `scripts/coop/ecoflap_wanda.sh`）：

```bash
cd /root/autodl-tmp/ECoFLaP/CoOp
pip install -r requirements.txt
```

**依赖很少：** ftfy, regex, tqdm（若已装 LAVIS，多数已存在）。

---

## 三、UPop（BLIP 对比实验）— 可选

README 要求：**Python 3.8**，**PyTorch 1.11**，**CUDA 11.3**。

### 方式 A：用 conda 从 environment 创建（可能因平台需微调）

```bash
cd /root/autodl-tmp/ECoFLaP/UPop
conda env create -f environment.yml
conda activate upop
```

### 方式 B：手动建环境（更可控）

```bash
conda create -n upop python=3.8 -y
conda activate upop
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio=0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
cd /root/autodl-tmp/ECoFLaP/UPop
pip install -r requirements.txt   # 若存在 pip 版；否则按 README 用 environment.yml
```

UPop 的 `requirements.txt` 在仓库里是 conda 导出格式，若没有 pip 版，就只用方式 A 或方式 B 的 conda + 再根据需要 `pip install` 缺的包。

---

## 四、LLaMA 实验 — 可选

单独环境，按 `LLaMA/INSTALL.md`：

```bash
conda create -n prune_llm python=3.9 -y
conda activate prune_llm
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install transformers==4.28.0 datasets==2.11.0 wandb sentencepiece
pip install accelerate==0.18.0
```

之后在 `ECoFLaP/LLaMA/` 下跑脚本。

---

## 总结：你「要装」的东西

| 要跑的模块 | 需要装的环境 |
|------------|----------------|
| **只跑 BLIP-2 / FlanT5 / ViT** | 一个 conda 环境 + PyTorch + `cd ECoFLaP/LAVIS && pip install -e .` |
| **还要跑 CLIP (CoOp)** | 同上 + `cd ECoFLaP/CoOp && pip install -r requirements.txt` |
| **还要跑 UPop (BLIP)** | 单独 UPop 环境（Python 3.8 + PyTorch 1.11）+ `conda env create -f UPop/environment.yml` 或按上面方式 B |
| **还要跑 LLaMA** | 单独 LLaMA 环境（Python 3.9 + PyTorch 1.10.1）+ INSTALL.md 里的 pip 包 |

**最小可运行（只跑主实验）：**

```bash
conda create -n ecoflap python=3.9 -y
conda activate ecoflap
pip install torch torchvision torchaudio
cd /root/autodl-tmp/ECoFLaP/LAVIS && pip install -e .
```

数据集需另外按各子目录的 `lavis/datasets/download_scripts/` 或 README 下载。
