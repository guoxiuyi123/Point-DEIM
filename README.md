# Point

本目录用于在不改动 DEIM 主工程的前提下，提供：

- NWPU-VHR-10 的 point-only / full-box baseline 配置
- 统一的训练/评测包装入口
- 结果汇总、表格生成与可视化脚本
- 项目指南（原理/复现/论文实验/排障）：[PROJECT_GUIDE.md](PROJECT_GUIDE.md)

## 目录结构

- configs/: 实验配置（YAML，基于 /home/pc/gxy/DEIM 的配置体系）
- scripts/: 结果汇总、表格与可视化工具
- train.py: 训练/评测包装入口（内部调用 DEIM/train.py）

## 快速开始

### 1) 准备 NWPU-VHR-10 COCO（在 Point 内落地）

```bash
python scripts/prepare_nwpu_vhr10_coco.py \
  --src-root /home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train \
  --out-root dataset/nwpu_vhr10 \
  --val-ratio 0.2 \
  --seed 0
```

### 2) 跑通 smoke（CPU 也可）

```bash
python train.py --preset nwpu_vhr10_full_box_smoke --seed 0
python train.py --preset nwpu_vhr10_point_only_smoke --seed 0
python train.py --preset nwpu_vhr10_point_only_smallobj_all_smoke --seed 0
```

### 3) 跑正式配置（单点监督 / 全监督上限）

```bash
python train.py --preset nwpu_vhr10_full_box --seed 0
python train.py --preset nwpu_vhr10_point_only --seed 0
python train.py --preset nwpu_vhr10_point_only_smallobj_all --seed 0
```

### 4) 评测与可视化

评测会在输出目录后缀追加 `_eval` 并写入 `pred_bbox.json`：

```bash
python train.py --preset nwpu_vhr10_full_box --seed 0 --test-only -r outputs/nwpu_vhr10_full_box/last.pth
python scripts/vis_pred.py \
  --pred-json outputs/nwpu_vhr10_full_box_eval/pred_bbox.json \
  --ann-file dataset/nwpu_vhr10/annotations/instances_val.json \
  --images-dir dataset/nwpu_vhr10/images/val \
  --out-dir outputs/vis_full_box
```

### 5) 结果汇总与表格

```bash
python scripts/make_table.py --outputs-dir outputs --out-md outputs/summary.md
```
