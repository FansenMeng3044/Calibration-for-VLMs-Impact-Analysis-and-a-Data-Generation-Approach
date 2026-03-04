# ECoFLaP 全部 Conda 环境安装要求与命令

以下仅列出**环境要求**和**安装命令**，不执行安装。按需选择对应实验（BLIP / LAVIS / CoOp / LLaMA）再执行。

---

## 1. UPop 环境（BLIP 实验：VQA / NLVR2 / Flickr / COCO Caption）

**用途**：BLIP 上的 ECoFLaP、Wanda、UPop 对比实验。

**环境名**：`upop`  
**要求**：
- Python 3.8.13
- PyTorch 1.11.0，CUDA 11.3.1，cuDNN 8.2.0
- torchvision 0.12.0，torchaudio 0.11.0
- 依赖见 `ECoFLaP/UPop/environment.yml`（含 conda + pip 列表）

**命令**（在项目根或 `UPop/` 下执行）：

```bash
cd /root/autodl-tmp/ECoFLaP/UPop
conda env create -f environment.yml
conda activate upop
```

说明：`environment.yml` 末尾的 `prefix` 可忽略，conda 会按当前环境路径创建。

---

## 2. LAVIS 环境（BLIP-2 / FlanT5 / ViT 实验）

**用途**：BLIP-2、FlanT5 XL、ViT 上的 ECoFLaP 与 Wanda 实验。

**环境名**：自定（如 `lavis`）  
**要求**：
- Python 3.8
- 其余依赖由 `pip install -e .` 或 `pip install salesforce-lavis` 拉取

**命令**：

```bash
conda create -n lavis python=3.8
conda activate lavis
cd /root/autodl-tmp/ECoFLaP/LAVIS
pip install -e .
# 或从 PyPI 安装：pip install salesforce-lavis
```

---

## 3. CoOp 环境（CLIP 实验）

**用途**：CLIP 上的 Wanda、SparseGPT、ECoFLaP (w/ Wanda)、ECoFLaP (w/ SparseGPT)。

**依赖**：先装 [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch)，再在 CoOp 下装 CLIP 等。

**环境名**：`dassl`（按 Dassl 官方）  
**要求**：
- Python 3.8
- PyTorch、torchvision（Dassl 示例用 cudatoolkit=10.2，可按需改 CUDA 版本）
- Dassl 的 requirements.txt + `python setup.py develop`
- CoOp 的 `requirements.txt`（CLIP 等）

**命令**：

```bash
# 3.1 安装 Dassl 环境（按 Dassl 官方文档）
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/
conda create -y -n dassl python=3.8
conda activate dassl
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch   # 或按需改 CUDA 版本
pip install -r requirements.txt
python setup.py develop
cd ..

# 3.2 在 Dassl 环境下装 CoOp 依赖
conda activate dassl
cd /root/autodl-tmp/ECoFLaP/CoOp
pip install -r requirements.txt
```

---

## 4. LLaMA 环境（LLaMA 剪枝实验）

**用途**：LLaMA 上的 ECoFLaP zeroth-order 实验。

**环境名**：`prune_llm`  
**要求**：
- Python 3.9
- PyTorch 1.10.1，torchvision 0.11.2，torchaudio 0.10.1，cudatoolkit 11.3
- transformers==4.28.0，datasets==2.11.0，wandb，sentencepiece
- accelerate==0.18.0

**命令**（见 `ECoFLaP/LLaMA/INSTALL.md`）：

```bash
conda create -n prune_llm python=3.9
conda activate prune_llm
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install transformers==4.28.0 datasets==2.11.0 wandb sentencepiece
pip install accelerate==0.18.0
```

说明：LLaMA tokenizer 与 transformers 存在已知问题，见 [transformers#22222](https://github.com/huggingface/transformers/issues/22222)，需按该 issue 建议处理。

---

## 汇总表

| 实验模块 | 环境名   | Python | PyTorch | CUDA   | 安装方式 |
|----------|----------|--------|---------|--------|----------|
| BLIP (UPop) | upop     | 3.8.13 | 1.11.0  | 11.3   | `conda env create -f UPop/environment.yml` |
| LAVIS    | lavis    | 3.8    | (随 LAVIS) | -   | `conda create -n lavis python=3.8` + `pip install -e .`（在 LAVIS/） |
| CoOp     | dassl    | 3.8    | (随 Dassl) | 10.2 示例 | 先装 Dassl，再 `pip install -r requirements.txt`（在 CoOp/） |
| LLaMA    | prune_llm| 3.9    | 1.10.1  | 11.3   | 见上面 LLaMA 命令 |

---

**说明**：四个环境互相独立，Python/PyTorch 版本不一致，不要混用同一 conda 环境跑不同模块。
