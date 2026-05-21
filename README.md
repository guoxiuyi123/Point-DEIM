# Point-DEIM

本目录用于在 Point-DEIM 内集成一份可运行的 DEIM 代码（见 `DEIM/`），并提供：

- NWPU-VHR-10 的 point-only / full-box baseline 配置
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
python train.py -c configs/nwpu_vhr10/deim_hgnetv2_n_point_only_smallobj_all_gated_v2_hgpre_smoke.yml --seed 0
```

### 2.5) （可选）启用 HGNetv2 stage1 预训练

已将 HGNetv2 stage1 权重同步到本目录：

- `weight/hgnetv2/PPHGNetV2_B0_stage1.pth`

使用预训练的 smoke / 正式训练：

```bash
python train.py -c configs/nwpu_vhr10/deim_hgnetv2_n_point_only_smallobj_all_gated_v2_hgpre_smoke.yml --seed 0
python train.py -c configs/nwpu_vhr10/deim_hgnetv2_n_point_only_smallobj_all_gated_v2_hgpre.yml --seed 0
```

### 3) 跑正式配置（单点监督 / 全监督上限）

```bash
python train.py -c configs/nwpu_vhr10/deim_hgnetv2_n_full_box.yml --seed 0
python train.py -c configs/nwpu_vhr10/deim_hgnetv2_n_point_only.yml --seed 0
python train.py -c configs/nwpu_vhr10/deim_hgnetv2_n_point_only_smallobj_all.yml --seed 0
```

### 4) 评测与可视化

评测会在输出目录后缀追加 `_eval` 并写入 `pred_bbox.json`：

```bash
python train.py -c configs/nwpu_vhr10/deim_hgnetv2_n_full_box.yml --seed 0 --test-only -r outputs/nwpu_vhr10_full_box/last.pth
python scripts/vis_pred.py \
  --pred-json outputs/nwpu_vhr10_full_box_eval/pred_bbox.json \
  --ann-file dataset/nwpu_vhr10/annotations/instances_val.json \
  --images-dir dataset/nwpu_vhr10/images/val \
  --out-dir outputs/vis_full_box
```

如需绕过 preset 直接用 YAML（等价于 train.py 内部调用），在本目录下运行：

```bash
python DEIM/train.py -c configs/nwpu_vhr10/deim_hgnetv2_n_point_only_smallobj_all_gated_v2_hgpre.yml --seed 0
python DEIM/train.py -c configs/nwpu_vhr10/deim_hgnetv2_n_point_only_smallobj_all_gated_v2_hgpre.yml --test-only -r outputs/nwpu_vhr10_point_only_smallobj_all_gated_v2_hgpre/best_stg1.pth
```

### 5) 结果汇总与表格

```bash
python scripts/make_table.py --outputs-dir outputs --out-md outputs/summary.md
```
