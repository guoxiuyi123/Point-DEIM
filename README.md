# Point-DEIM

本目录用于在 Point-DEIM 内集成一份可运行的 DEIM 代码（见 `DEIM/`），并提供：

- VisDrone2019 的 point-only / full-box baseline 配置（默认主线）
- NWPU-VHR-10 的 point-only / full-box baseline 配置（legacy，仅保留兼容）
- 统一的训练/评测包装入口
- 结果汇总、表格生成与可视化脚本
- 项目指南：待补充

## 目录结构

- DEIM/: vendored DEIM 主工程代码（Point-DEIM 可独立运行，不依赖外部 /home/pc/gxy/DEIM）
- configs/: 实验配置（YAML，基于 `DEIM/configs` 的配置体系）
- scripts/: 结果汇总、表格与可视化工具
- train.py: 训练/评测包装入口（推荐使用，内部调用 `DEIM/train.py`）
- DEIM/train.py: vendored DEIM 的训练/评测入口（需要时可直接调用）

## 快速开始

### 1) 准备 VisDrone2019（point-only，COCO）

```bash
python scripts/prepare_visdrone2019_coco.py \
  --src-root /home/pc/gxy/dataset/VisDrone2019/VisDrone2019 \
  --out-root dataset/visdrone2019
```

### 2) 跑通 smoke（CPU 也可）

```bash
python train.py -c configs/visdrone2019/deim_hgnetv2_n_point_only_ptteacher_smoke.yml --seed 0
```

### 2.5) （可选）启用 HGNetv2 stage1 预训练

已将 HGNetv2 stage1 权重同步到本目录：

- `weight/hgnetv2/PPHGNetV2_B0_stage1.pth`

使用预训练的 smoke / 正式训练：

```bash
python train.py -c configs/visdrone2019/deim_hgnetv2_n_point_only_ptteacher_smoke.yml --seed 0
python train.py -c configs/visdrone2019/deim_hgnetv2_n_point_only_ptteacher.yml --seed 0
```

### 3) 跑正式配置（point-only / 全监督上限）

```bash
python train.py -c configs/visdrone2019/deim_hgnetv2_n_point_only_ptteacher.yml --seed 0
python train.py -c configs/visdrone2019/deim_hgnetv2_n_full_box.yml --seed 0
```

### 4) 评测与可视化

评测会在输出目录后缀追加 `_eval` 并写入 `pred_bbox.json`：

`--test-only` 评测完成后会在同目录生成 `tide_result/`（TIDE 指标与可视化；若图像生成失败，会留下 `plot_error.txt`）。

```bash
python train.py -c configs/visdrone2019/deim_hgnetv2_n_point_only_ptteacher.yml --seed 0 --test-only -r outputs/visdrone2019_point_only_ptteacher/last.pth
python scripts/vis_pred.py \
  --pred-json outputs/visdrone2019_point_only_ptteacher_eval/pred_bbox.json \
  --ann-file dataset/visdrone2019/annotations/instances_val.json \
  --images-dir dataset/visdrone2019/images/val \
  --out-dir outputs/vis_visdrone2019
```

如需绕过 preset 直接用 YAML（等价于 train.py 内部调用），在本目录下运行：

```bash
python DEIM/train.py -c configs/visdrone2019/deim_hgnetv2_n_point_only_ptteacher.yml --seed 0
python DEIM/train.py -c configs/visdrone2019/deim_hgnetv2_n_point_only_ptteacher.yml --test-only -r outputs/visdrone2019_point_only_ptteacher/last.pth
```

### 5) 结果汇总与表格

```bash
python scripts/make_table.py --outputs-dir outputs --out-md outputs/summary.md
```

## Legacy：NWPU-VHR-10

NWPU-VHR-10 的配置与数据准备脚本仍可使用，但本仓库的默认复现路径与后续迭代以 VisDrone2019 为主线。

### 1) 准备 NWPU-VHR-10（legacy，COCO）

```bash
python scripts/prepare_nwpu_vhr10_coco.py \
  --src-root /home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train \
  --out-root dataset/nwpu_vhr10
```

### 2) 训练 / `--test-only`

```bash
python train.py -c configs/nwpu_vhr10/deim_hgnetv2_n_point_only.yml --seed 0
python train.py -c configs/nwpu_vhr10/deim_hgnetv2_n_point_only.yml --seed 0 --test-only -r outputs/nwpu_vhr10_point_only_fullscale/last.pth
```
