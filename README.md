# UIC_TES

**Ultrasound Image Classification · Training & Evaluation System**

基于 PyTorch 的医学超声图像分类实验框架，支持多种骨干网络、训练策略与数据集，面向子宫病灶、皮肤超声等医学影像分类任务。

---

## 项目简介

UIC_TES 提供从数据组织、K 折划分、模型训练到指标评估与可视化解释的完整流程，便于在统一代码基座上对比不同网络结构与增强策略。

主要能力包括：

- **多数据集**：子宫超声（Uterus）、皮肤超声（Dermatologic Ultrasound）、BUSI、肾脏超声、AUITD 等
- **多训练模式**：Baseline / SLA / SLA+SD / FASD
- **多骨干网络**：ResNet、VGG、DenseNet、ViT、Swin、CMT、DaViT 等
- **完整评估**：Accuracy、AUC（整体与各类别）、混淆矩阵；支持 checkpoint 保存
- **可解释性**：CAM、ViT Attention Rollout / Grad-Rollout 等可视化工具

---

## 目录结构

```text
UIC_TES/
├── main.py                 # 训练入口与命令行参数
├── trainer.py              # 各训练模式下的 train/test 流程
├── models/                 # 模型实现与第三方参考代码
│   ├── Uterus/             # 主用骨干网络集合
│   ├── FASD_main/
│   ├── EKD_FWSNet_main/
│   ├── ODConv_main/
│   └── tmi2022_main/
├── SLA/                    # Self-supervised / SLA 相关增强与训练
├── utils/                  # 数据加载、CSV 生成、评估、可视化
├── Dermatologic_Ultrasound/# 皮肤超声相关数据元信息
└── tool_new/               # 辅助脚本与工具
```

---

## 环境要求

- Python ≥ 3.7
- PyTorch（建议带 CUDA）
- 常用依赖：`numpy`、`pandas`、`scikit-learn`、`Pillow`、`optuna` 等

```bash
pip install torch torchvision numpy pandas scikit-learn pillow optuna
```

---

## 快速开始

### 1. 准备数据

将图像按类别组织到数据集目录，并按需修改 `main.py` 中的 `--dataset-path` 与 `--checkpoint`。

可使用 `utils/` 下脚本生成 CSV / 划分 K 折，例如：

- `utils/getCsv_Dermatologic_Ultrasound.py`
- `utils/getCsv_BUSI.py`
- `utils/getCsv_kidney.py`
- `utils/getCsv_auitd.py`

### 2. 启动训练

`--mode` 为必填项，常用取值示例：`DLB`（baseline）、`SLA`、`SLA_SD`、`FASD` 等（以代码中实际分支为准）。

```bash
python main.py \
  --mode DLB \
  --arch resnet18 \
  --dataset Uterus \
  --dataset-path /path/to/your/dataset \
  --checkpoint /path/to/checkpoint \
  --epochs 300 \
  --train-batch 128 \
  --test-batch 100 \
  --lr 0.1 \
  --gpu-id 0
```

SLA 相关示例：

```bash
python main.py \
  --mode SLA_SD \
  --arch resnet18 \
  --aug rotation \
  --T 1.0 \
  --gpu-id 0
```

### 3. 评估与可视化

- 训练过程会记录 Loss / Acc / AUC，并在收敛或最优时保存混淆矩阵与 checkpoint
- 可视化与解释可参考 `utils/cam.py`、`utils/vit_explain.py` 等模块

---

## 训练模式说明

| 模式 | 说明 |
|------|------|
| Baseline (`DLB` 等) | 标准监督分类训练 |
| SLA | 结合自监督式增强的聚合训练 |
| SLA_SD | SLA + 单图 / 聚合双分支联合优化 |
| FASD | FASD 相关训练策略 |

评价指标同时支持 **单图（single）** 与 **统计/聚合（statis / agg）** 两种粒度。

---

## 注意事项

1. 默认数据与 checkpoint 路径为服务器本地路径，请按本机环境修改后再运行。
2. 仓库中含部分实验缓存（如 `__pycache__`、`.idea`），不影响核心训练逻辑。
3. `models/` 下部分子目录为论文复现 / 参考实现，可按需选用。

---

## License

本项目仅供学习与研究使用。若引用其中第三方模型代码，请遵循对应原仓库许可协议。
