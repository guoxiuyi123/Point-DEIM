# Point 项目指南（原理 / 复现 / 论文实验 / 排障）

本文面向在不改动 [DEIM](../DEIM) 主工程的前提下，复现并扩展本仓库的“点监督（point-only）”检测实验。Point 目录本质上是一个轻量的实验壳层：提供配置、包装入口、点监督扩展与结果汇总脚本。

## 1. Point 相对 DEIM 的定位

- DEIM：训练/评测主框架（模型、数据管线、solver、日志、评测等）。
- Point：在仓库根目录新增一个“旁路工程”，通过 Python import 注入方式：
  - 注册点监督 criterion 与小目标增强 transform
  - 提供 NWPU-VHR-10 的 point-only / full-box 配置
  - 提供统一的训练/评测入口与汇总可视化脚本

这种结构的好处：

- 不改 DEIM 代码即可做点监督实验（便于同步/对齐上游）。
- Point 目录可随时拆掉/迁移，不污染 DEIM 的默认行为。

## 2. 目录结构与关键入口

- [train.py](train.py)：包装入口，负责选择 preset/config，并把参数透传给 DEIM 的 `train.py`
- [deim_entry.py](deim_entry.py)：真正的“注入入口”
  - 把 `Point/`、`DEIM/`、仓库根目录加入 `sys.path`
  - `import Point.point_ext` 触发扩展注册
  - `runpy.run_path(DEIM/train.py)` 进入 DEIM 主流程
- [point_ext/](point_ext/)
  - [point_sup_criterion.py](point_ext/point_sup_criterion.py)：点监督核心 criterion（loss_point + 伪框记忆）
  - [small_object.py](point_ext/small_object.py)：小目标增强三件套（裁剪/尺度初始化/阈值调度）
- [configs/](configs/)：实验 YAML（引用 DEIM 的 base config）
- [scripts/](scripts/)：数据准备、评测可视化、汇总表格
- [pycocoeval/](pycocoeval/)：为 DEIM 提供 `pycocoeval` 模块（避免额外编译依赖）

## 3. 运行链路（从 train.py 到 DEIM）

1. `python train.py --preset ...` 或 `python train.py --config ...`
2. Point/train.py 组装命令：`python deim_entry.py -c <config> [--seed ...] <透传参数>`
3. deim_entry.py 在运行 DEIM/train.py 前先 `import Point.point_ext`
4. point_ext 中的 `@register()` 会把扩展类注册到 DEIM 的 registry（例如 criterion、transform）
5. DEIM 按 YAML 里写的 `criterion: PointSupDEIMCriterionV2` 等名称，从 registry 里取到扩展实现

参数透传说明：

- Point/train.py 使用 `parse_known_args()`，因此 `--test-only -r ... --device ...` 等 DEIM 参数可直接跟在后面
- 如需显式分隔，也可以用 `--`：`python train.py --preset xxx -- --test-only -r ...`

## 4. 点监督核心机制（Point → 伪框记忆 → 匹配 → 更新 → loss_point）

点监督训练的数据与监督信号大致分两层：

- 数据层：训练集使用 `CocoPointDetection`，从 COCO 格式中读取点标注（`targets[*]["points"]`），并可选择丢弃真实框（`drop_boxes: True`）
- 损失层：用“伪框”把点监督转成 DEIM 可训练的 boxes 监督，同时额外加入 `loss_point` 约束预测框中心

### 4.1 伪框记忆（PseudoBoxMemory）

实现位置：

- [PointSupDEIMCriterionV2](point_ext/point_sup_criterion.py) 中构造 `PseudoBoxMemory`

每张图像（以 `targets[*]["idx"]` 作为 sample id）维护一份伪框集合：

1. 初始化：用点生成初始伪框（中心为点，宽高为先验 `prior_wh`，并受 `min_wh/max_wh` 约束）
2. 训练时：把 `PseudoBoxMemory.get()` 得到的伪框写入 `targets["boxes"]`，喂给 DEIM 原有的匹配与回归损失

### 4.2 匹配与更新（HungarianMatcher + memory.update）

在 `forward()` 中：

- 先用伪框跑一遍 DEIMCriterion 的损失（分类、回归、GIoU 等）
- 再用 matcher 在“预测 ↔ 伪框”之间做一次匹配，取出 matched 的预测框与分数
- 将 matched 的预测框写回 `PseudoBoxMemory.update()`，不断“自举”伪框质量

分数（score）计算有 warmup：

- warmup 阶段可按类无关方式取 `max prob`
- 之后按匹配到的 label 取对应类别概率

### 4.3 loss_point（约束中心到点）

`loss_point` 直接对 matched 预测框的 `(cx, cy)` 与目标点做 Smooth L1：

- 位置：[loss_points](point_ext/point_sup_criterion.py)
- 作用：提供“中心对齐”信号，降低仅靠伪框回归导致的漂移

训练时可关注日志打印的匹配覆盖率（matched / points），覆盖率过低往往意味着阈值、初始化尺度或数据/点质量存在问题。

## 5. 小目标三件套（对应哪些配置）

Point 目录提供三类增强，既可单独开，也可组合：

### 5.1 点引导裁剪（PointGuidedCrop）

- 位置：[PointGuidedCrop](point_ext/small_object.py)
- 思路：优先围绕更小的目标框做 crop，提高小目标在网络输入中的占比
- 配置入口：在 YAML 的 `transforms.ops` 里加入 `PointGuidedCrop`
  - 例子：[deim_hgnetv2_n_point_only_pg_crop.yml](configs/nwpu_vhr10/deim_hgnetv2_n_point_only_pg_crop.yml)

### 5.2 尺度自适应初始化（ScaleAdaptiveInit）

- 位置：`PointSupDEIMCriterionScaleAdaptiveInit`（在 [small_object.py](point_ext/small_object.py) 注册）
- 思路：对同图多点时，用 kNN 点间距离估计目标尺度，替代固定 `prior_wh`
- 配置入口：criterion 切到 `PointSupDEIMCriterionScaleAdaptiveInit`
  - 例子：[deim_hgnetv2_n_point_only_sa_init.yml](configs/nwpu_vhr10/deim_hgnetv2_n_point_only_sa_init.yml)

### 5.3 score_thresh 调度（ScoreThreshSchedule）

- 位置：`PointSupDEIMCriterionScoreThreshSchedule`（在 [small_object.py](point_ext/small_object.py) 注册）
- 思路：早期阈值更高（更“苛刻”更新），后期阈值降低（提升 recall），减少伪框更新噪声
- 配置入口：criterion 切到 `PointSupDEIMCriterionScoreThreshSchedule`
  - 例子：[deim_hgnetv2_n_point_only_score_sched.yml](configs/nwpu_vhr10/deim_hgnetv2_n_point_only_score_sched.yml)

### 5.4 全量组合（smallobj_all）

- 例子：[deim_hgnetv2_n_point_only_smallobj_all.yml](configs/nwpu_vhr10/deim_hgnetv2_n_point_only_smallobj_all.yml)
- 组合内容：PointGuidedCrop + 尺度自适应初始化 + 阈值调度（并可调 pseudo_box 的 `center_radius/score_thresh` 等）

## 6. 复现流程（端到端）

以下命令均建议在 `Point/` 目录下执行（Point/train.py 会自动把工作目录设为 Point 根）。

### 6.1 数据准备：NWPU-VHR-10 → COCO（落地到 Point/dataset）

```bash
python scripts/prepare_nwpu_vhr10_coco.py \
  --src-root /home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train \
  --out-root dataset/nwpu_vhr10 \
  --val-ratio 0.2 \
  --seed 0
```

输出（关键路径）：

- `dataset/nwpu_vhr10/images/{train,val}/`
- `dataset/nwpu_vhr10/annotations/instances_{train,val}.json`
- `dataset/nwpu_vhr10/annotations/instances_{train,val}_smoke.json`（用于快速 smoke，默认会额外生成；如需关闭可在脚本中将 `--smoke-train/--smoke-val` 设为 `<=0`）

### 6.2 训练：full-box 上限 vs point-only

建议先跑 smoke（CPU 也可）确认链路无误：

```bash
python train.py --preset nwpu_vhr10_full_box_smoke --seed 0
python train.py --preset nwpu_vhr10_point_only_smoke --seed 0
python train.py --preset nwpu_vhr10_point_only_smallobj_all_smoke --seed 0
```

full-box：

```bash
python train.py --preset nwpu_vhr10_full_box --seed 0
```

point-only：

```bash
python train.py --preset nwpu_vhr10_point_only --seed 0
```

### 6.3 评测：test-only + 生成 pred_bbox.json

```bash
python train.py --preset nwpu_vhr10_full_box --seed 0 --test-only -r outputs/nwpu_vhr10_full_box/last.pth
python train.py --preset nwpu_vhr10_point_only --seed 0 --test-only -r outputs/nwpu_vhr10_point_only/last.pth
```

常见输出（以 `*_eval/` 为后缀）：

- `pred_bbox.json`：COCO detection 预测
- `tide_result/`：TIDE 分析图（若环境中启用）
- `eval.pth`：评测阶段保存的权重/中间产物（如启用）

训练/评测的 TensorBoard 日志通常位于：

- `outputs/<exp>/summary/events.out.tfevents.*`
- `outputs/<exp>_eval/summary/events.out.tfevents.*`

可用如下方式查看：

```bash
tensorboard --logdir outputs --port 6006
```

### 6.4 可视化：画 GT 与预测框

```bash
python scripts/vis_pred.py \
  --pred-json outputs/nwpu_vhr10_full_box_eval/pred_bbox.json \
  --ann-file dataset/nwpu_vhr10/annotations/instances_val.json \
  --images-dir dataset/nwpu_vhr10/images/val \
  --out-dir outputs/vis_full_box
```

### 6.5 结果汇总：从 log.txt 生成表格

```bash
python scripts/make_table.py --outputs-dir outputs --out-md outputs/summary.md
```

说明：

- `outputs/**/log.txt` 是 JSONL（每行一个 dict），脚本会按 AP 选择 best epoch 汇总
- `outputs/**/args.json` 可辅助记录每次运行的参数

## 7. 论文实验（对照 / 消融 / 组合）

本目录提供了“可写进论文表格”的最小实验矩阵，建议至少包含：

### 7.1 对照组（Upper Bound / Lower Bound）

- Upper bound（全监督上限）：`nwpu_vhr10_full_box`
- Lower bound（点监督 baseline）：`nwpu_vhr10_point_only`

### 7.2 消融（逐个打开小目标组件）

- + 尺度自适应初始化：`nwpu_vhr10_point_only_sa_init`
- + 阈值调度：`nwpu_vhr10_point_only_score_sched`
- + 点引导裁剪：`nwpu_vhr10_point_only_pg_crop`

### 7.3 组合（最终配置）

- `nwpu_vhr10_point_only_smallobj_all`

### 7.4 多 seed 建议与汇总

建议每个配置至少 3 个 seed，例如 0/1/2：

```bash
for s in 0 1 2; do
  python train.py --preset nwpu_vhr10_point_only_smallobj_all --seed $s
done
python scripts/make_table.py --outputs-dir outputs --out-md outputs/summary.md
```

如需更严格统计（均值/方差、置信区间），可在 `outputs/summary.md` 的基础上二次处理，或扩展 `scripts/collect_results.py`。

## 8. 排障（按现象分组）

### 8.1 找不到/导入不到扩展类（criterion 或 transform）

现象：

- 启动时报 `Unknown criterion: PointSupDEIMCriterionV2` / registry 找不到类

排查：

- 确认用的是 Point 的入口：`python Point/train.py ...`，不要直接跑 `DEIM/train.py`
- 确认 [deim_entry.py](deim_entry.py) 中 `import Point.point_ext` 未被删除
- 确认 YAML 里的 `criterion:` 名称与注册类名一致（大小写完全一致）

### 8.2 COCO 数据路径/标注错误

现象：

- `FileNotFoundError: dataset/...`
- `KeyError: images/annotations/categories`
- 评测阶段 AP 全为 0 或直接报错

排查：

- 用 `scripts/prepare_nwpu_vhr10_coco.py` 重新生成 COCO，并确认 `images/{train,val}` 与 `instances_{train,val}.json` 存在
- 确认 YAML 中 `img_folder`/`ann_file` 为 Point 目录下的相对路径（不要写成 DEIM 目录）

### 8.3 点监督训练不收敛 / 匹配覆盖率低

现象：

- 日志中 `[PointSup] ... ratio=...` 长期很低（接近 0）
- loss_point 非常大或震荡，AP 不提升

排查思路（从易到难）：

- 数据是否确实提供点：`targets["points"]` 是否为空（点数量是否和目标数一致）
- 初始伪框尺度：调 `prior_wh / min_wh / max_wh` 或改用 `nwpu_vhr10_point_only_sa_init`
- 伪框更新阈值：调 `pseudo_box.score_thresh` 或用 `nwpu_vhr10_point_only_score_sched`
- 小目标场景：优先启用 `nwpu_vhr10_point_only_pg_crop` 或 `smallobj_all`

### 8.4 评测阶段报 pycocoeval 相关错误

现象：

- `ModuleNotFoundError: pycocoeval`

排查：

- Point 目录自带 [pycocoeval/](pycocoeval/) 模块，只有通过 Point/deim_entry 注入 `sys.path` 才能被 DEIM 找到
- 若你改动了启动方式，确保 `Point/` 在 `PYTHONPATH` 或 `sys.path` 前列

### 8.5 OpenCV / 可视化脚本报错

现象：

- `import cv2` 失败或写图失败

排查：

- `scripts/vis_pred.py` 依赖 OpenCV；如仅需要训练/评测，可跳过该脚本
- 如果需要可视化，补齐运行环境的 OpenCV 依赖后重试
