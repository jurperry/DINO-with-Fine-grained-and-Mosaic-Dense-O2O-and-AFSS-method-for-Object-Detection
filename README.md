# DINO-with-Fine-grained-and-Mosaic-Dense-O2O-and-AFSS-method-for-Object-Detection
The new model introduces the gated attention mechanism to Multi-Scale Deformable Attention, we can use pre norm method and large learning rate, improve the training stability and performance.Use a new mosaic augmentation to improve the training performance, by introduce Dense O2O(Like DEIM). And AFSS(Anti Forgetting Sampling Strategy) is a way to training much faster than before without losing performance.

# Training(直接训练)
cd project
python train.py --config configs\coco_tiny.yaml --epochs 50 -bs 2 -nc 80 -nq 300 -lr 2e-4 -lr_backbone 2e-5 -gate_attn -pm

# Resuming(中途断点续训)
cd project
python train.py --config configs\coco_tiny.yaml --epochs 50 -bs 2 -nc 80 -nq 300 -lr 2e-4 -lr_backbone 2e-5 -gate_attn -pm
--resume "results\your_configs\last_checkpoint.pth"
# Tuning(预训练微调)
python train.py --config configs\coco_tiny.yaml --epochs 50 -bs 2 -nc 80 -nq 300 -lr 2e-4 -lr_backbone 2e-5 -gate_attn -pm
# --tuning "results\best_checkpoint.pth"
