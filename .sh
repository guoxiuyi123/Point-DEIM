CUDA_VISIBLE_DEVICES=0 python train.py \
  -c configs/deimv2/deimv2_dinov3_m_coco.yml \
  --use-amp \
  --seed 0 \
  -u train_dataloader.total_batch_size=2

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  -c configs/deimv2/deimv2_dinov3_m_coco.yml \
  --use-amp \
  --seed 0 \
  -u train_dataloader.total_batch_size=8

python train.py -c configs/deimv2/deimv2_dinov3_m_coco.yml --test-only -r outputs/deimv2_dinov3_m_coco/best_stg1.pth

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  -c /home/pc/gxy/Point-DEIM_baseline/configs/yaml/deim_dfine_hgnetv2_n_mg_visdrone_point_teacher.yml \
  --use-amp \
  --seed 0 \
  -u train_dataloader.total_batch_size=32
