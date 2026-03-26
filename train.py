# yaml, argparse是常用的文件配置与指令输入形式, 可以实现配置文件与实际代码分离
# 用户在不修改代码的情况下, 通过指令接口和配置文件就可以实现修改参数, 简化使用流程
import yaml
import argparse
import csv
import os
from datetime import datetime

import random
import numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import TrainDataset_for_DETR, AFSSManager
from models import compute_train_loss, GatedAttention_FineGrained_DINO_Swin
from utils import collate_fn, build_optimizer_and_scheduler, get_total_grad_norm, ModelEMA
from val import validate_model, printer_eval
from tools import empty_filter

# coco_tiny 80 class 500 train 500val
# python train.py --config configs\coco_tiny.yaml --epochs 50 -bs 2 -nc 80 -nq 300 -lr 2e-4 -lr_backbone 2e-5 -gate_attn -pm
# --resume "E:\python_project\paper_project\AFSS_DINO\results\coco_tiny\last_checkpoint.pth"
# --tuning "E:\python_project\paper_project\gated_fine_dino\results\pretrained\best_checkpoint.pth"

def set_seed(seed=42):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Contrastive Denoising DETR Training with YAML Config")
    parser.add_argument("--config", "-c", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, help="Override training epochs")
    parser.add_argument("--batch_size", "-bs", type=int, help="Override batch size")
    parser.add_argument("--learning_rate", "-lr", type=float, help="Override learning rate")
    parser.add_argument("--learning_rate_backbone", "-lr_backbone", type=float, help="Override backbone's learning rate")
    parser.add_argument("--learning_rate_proj_mult", "-lr_linear_proj_mult", type=float, help="Override projection's learning rate")
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", "-nw", type=int, help="Override number of workers")
    parser.add_argument("--num_classes", "-nc", type=int, help="Override number of classes")
    parser.add_argument("--num_queries", "-nq", type=int, help="Override number of queries")
    parser.add_argument("--mosaic_prob", "-mosaic", type=float, help="Override mosaic_probability")
    # 是否使用pin_memory加速, 输入--pin_memory才是True, 不加指令就是默认False
    parser.add_argument("--pin_memory", "-pm", action="store_true", help="Override pin_memory")
    # 是否开启门控注意力, 在指令后面跟 -gate_attn就变成开启True, 不加指令就是默认False
    parser.add_argument("--gate_attention", "-gate_attn", action="store_true", help="Use gated attention. Default is False.")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Override prefetch_factor")
    # 是否冻结backbone
    parser.add_argument("--freeze_backbone", action="store_true", help="Backbone freeze. Default is False.")
    # AFSS更新间隔轮数
    parser.add_argument("--afss_epochs", type=int, default=5, help="Override AFSS epochs. Default is 5.")
    
    # for loss compute
    # if use MAL, alpha=1.0, gamma=1.5, if use VFL, alpha=0.75, gamma=2.0
    parser.add_argument("--cls_loss_method", type=str, default='mal', help="Loss method")
    parser.add_argument("--alpha", type=float, default=1.0, help="Loss hyperparameter alpha")
    parser.add_argument("--gamma", type=float, default=1.5, help="Loss hyperparameter gamma")
    
    # for optimizer
    parser.add_argument("--optimizer", type=str, default='adamw', help="Override optimizer")

    # for ema model
    parser.add_argument("--ema_decay", type=float, help="Override ema decay")
    parser.add_argument("--ema_warmups", type=int, help="Override ema warmups")
    parser.add_argument("--ema_start", type=int, help="Override ema start")

    # 验证时使用ema模型还是主模型
    parser.add_argument("--use_ema", type=bool, help="Use EMA model. Default is False.")
    
    # for device choosing
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Force device selection")

    # for resume and tuning training
    parser.add_argument("--resume", "-r", type=str, default=None, help=f"Path to the checkpoint to resume training from. "
                                                                    f"If provided, loads model weights, optimizer state, scheduler state, and epoch count.")
    parser.add_argument("--tuning", "-t", type=str, default=None, help="Tuning model from checkpoint. "
                                                                    f"If provided, loads model weights and freezes all layers except the final classification layer.")

    return parser.parse_args()

def load_config(config_path, args):
    # 执行config_path命令
    # args是命令指定, config是配置文件指定
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 命令行参数覆盖配置
    if args.epochs is not None: 
        config['training']['epochs'] = args.epochs
    if args.learning_rate is not None: 
        config['training']['learning_rate'] = args.learning_rate
    if args.learning_rate_backbone is not None: 
        config['training']['learning_rate_backbone'] = args.learning_rate_backbone
    if args.learning_rate_proj_mult is not None: 
        config['training']['learning_rate_proj_mult'] = args.learning_rate_proj_mult
    if args.batch_size is not None: 
        config['training']['batch_size'] = args.batch_size
    if args.clip_max_norm is not None: 
        config['training']['clip_max_norm'] = args.clip_max_norm
    if args.seed: config['training']['seed'] = args.seed
    if args.num_workers is not None: 
        config['training']['num_workers'] = args.num_workers
    if args.num_classes is not None: 
        config['training']['num_classes'] = args.num_classes
    if args.num_queries is not None: 
        config['training']['num_queries'] = args.num_queries
    if args.mosaic_prob is not None: 
        config['training']['mosaic_prob'] = args.mosaic_prob
    # 总是写入pin_memory的值
    config['training']['pin_memory'] = args.pin_memory
    # 总是写入gate_attn的值
    config['training']['gate_attention'] = args.gate_attention
    # 总是写入freeze_backbone的值
    config['training']['freeze_backbone'] = args.freeze_backbone
    # 总是写入afss_epochs的值
    config['training']['afss_epochs'] = args.afss_epochs

    if args.cls_loss_method: config['training']['cls_loss_method'] = args.cls_loss_method
    if args.alpha: config['training']['alpha'] = args.alpha
    if args.gamma: config['training']['gamma'] = args.gamma
    # optimizer
    if args.optimizer: config['training']['optimizer'] = args.optimizer
    if args.prefetch_factor: config['training']['prefetch_factor'] = args.prefetch_factor
    
    if args.ema_decay: config['training']['ema_decay'] = args.ema_decay
    if args.ema_warmups: config['training']['ema_warmups'] = args.ema_warmups
    if args.ema_start: config['training']['ema_start'] = args.ema_start
    if args.use_ema: config['training']['use_ema'] = args.use_ema

    if args.device: config['training']['device'] = args.device

    # 总是写入resume和tuning的值
    config['training']['resume'] = args.resume
    config['training']['tuning'] = args.tuning
    return config

def main(config, device_override=None):
    # device choosing
    device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # path configs
    paths = config['paths']
    resume_path = config['training']['resume']
    tuning_path = config['training']['tuning']
    # training configs
    train_config = config['training']
    
    # lr config
    base_lr = train_config["learning_rate"]
    lr_backbone = train_config["learning_rate_backbone"]
    lr_linear_proj_mult = train_config['learning_rate_proj_mult']

    # loss hyperparameter
    cls_loss_method = train_config['cls_loss_method']
    alpha = train_config['alpha']
    gamma = train_config['gamma']
    # clip_max_norm
    max_norm = train_config['clip_max_norm']
    if train_config['num_workers'] > 0:
        prefetch_factor = train_config['prefetch_factor']
    else:
        prefetch_factor = None
    # afss epochs for update
    afss_epochs = train_config['afss_epochs']
    
    print("Training configs:")
    print("="*100)
    for k, v in paths.items():
        print(f"{k}: {v}")
    print(f"resume: {resume_path}")
    print(f"tuning: {tuning_path}")
    print("-"*100)
    for k1, v1 in train_config.items():
        if k1 not in ['resume', 'tuning']:
            print(f"{k1}: {v1}")
    print("="*100 + "\n")

    # fix the seed for reproducibility
    set_seed(train_config['seed'])
    g = torch.Generator() # 用于数据集的生成随机种子

    # model initialization
    dino_model = GatedAttention_FineGrained_DINO_Swin(
        num_queries=train_config['num_queries'],
        num_classes=train_config['num_classes'],
        gate_attn=train_config['gate_attention'],
        freeze_backbone=train_config['freeze_backbone'],
    ).to(device)

    # 初始化EMA - 添加EMA配置参数到训练配置中
    ema_decay = train_config.get('ema_decay', 0.9999)  # 添加默认值
    ema_warmups = train_config.get('ema_warmups', 1000)  # 添加默认值
    ema_start = train_config.get('ema_start', 0)  # 添加默认值
    use_ema = train_config.get('use_ema', True) # 添加默认值
    
    # 创建EMA对象
    ema = ModelEMA(dino_model, decay=ema_decay, warmups=ema_warmups, start=ema_start)
    print(f"EMA initialized with decay={ema_decay}, warmups={ema_warmups}, start={ema_start}")

    # optimizer and lr_scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(
        dino_model, 
        lr=base_lr,
        lr_backbone=lr_backbone,
        lr_linear_proj_mult=lr_linear_proj_mult,
        weight_decay=train_config['weight_decay'],
        optimizer_type=train_config['optimizer'], 
        warmup_epochs=train_config['warmup_epochs'],
        final_finetune_epochs=train_config['final_finetune_epochs'],
        total_epochs=train_config['epochs'],
        mosaic_ratio=train_config['mosaic_ratio']
    )
    # 训练和验证的数据集准备
    train_imgdir = paths['train_imgdir']
    train_txtdir = paths['train_txtdir']
    val_imgdir = paths['val_imgdir']
    val_txtdir = paths['val_txtdir']

    # 检查训练集空标注问题
    src_img_dir = train_imgdir # 源图片文件夹
    src_txt_dir = train_txtdir   # 源TXT文件夹
    dst_img_dir = train_imgdir + '_filtered'
    dst_txt_dir = train_txtdir + '_filtered'   # 目标TXT文件夹
    two_levels_up = os.path.dirname(os.path.dirname(src_img_dir))
    empty_list_file = os.path.join(two_levels_up, "empty_files_train.txt") # 需要生成的空文件txt路径
    bool_check = empty_filter(src_img_dir, src_txt_dir, dst_img_dir, dst_txt_dir, empty_list_file)
    if bool_check == 1:
        train_imgdir = dst_img_dir
        train_txtdir = dst_txt_dir
    else:
        train_imgdir = src_img_dir
        train_txtdir = src_txt_dir
    # 检查验证集空标注问题
    val_src_img_dir = val_imgdir # 源图片文件夹
    val_src_txt_dir = val_txtdir   # 源TXT文件夹
    val_dst_img_dir = val_imgdir + '_filtered'
    val_dst_txt_dir = val_txtdir + '_filtered'   # 目标TXT文件夹
    val_empty_list_file = os.path.join(two_levels_up, "empty_files_val.txt") # 需要生成的空文件txt路径
    val_bool_check = empty_filter(val_src_img_dir, val_src_txt_dir, val_dst_img_dir, val_dst_txt_dir, val_empty_list_file)  
    if val_bool_check == 1:
        val_imgdir = val_dst_img_dir
        val_txtdir = val_dst_txt_dir
    else:
        val_imgdir = val_src_img_dir
        val_txtdir = val_src_txt_dir
    
    # Training dataset
    train_dataset = TrainDataset_for_DETR(
        imgdir_path=train_imgdir,
        txtdir_path=train_txtdir,
        image_set='train',
        mosaic_ratio=train_config['mosaic_ratio'], 
        mosaic_prob=train_config['mosaic_prob'],
        total_epochs=train_config['epochs'], 
        warmup_epochs=train_config['warmup_epochs'], 
        final_finetune_epochs=train_config['final_finetune_epochs'],
        scales=train_config['scales'], 
        max_size=train_config['max_size'], 
        val_size=train_config['val_size'], 
    )
    
    # 初始化 AFSS
    afss_manager = AFSSManager(num_samples=len(train_dataset.original_img_lst))
    # AFSS 预热轮次
    afss_warmup_epochs = 0

    start_epoch = 0 # 默认从第0个epoch开始
    best_map = 0.0 # 初始化记录的最佳mAP@50~95

    # for resume training
    if resume_path is not None:
        if os.path.isfile(resume_path):
            print(f"Loading checkpoint for resuming training: \n   {resume_path}...")
            checkpoint = torch.load(resume_path, map_location=device)
            # 加载模型权重 (主模型和EMA模型)
            dino_model.load_state_dict(checkpoint['model_state_dict']) # 加载主模型的权重
            ema.load_state_dict(checkpoint['ema_state_dict']) # 加载EMA模型权重与对应状态
            # 先加载优化器状态optimizer, 再加载调度器状态scheduler
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 在构建scheduler后调用
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # 加载开始的epoch (通常保存的是已完成的最后一个epoch)
            start_epoch = checkpoint.get('epoch', 0) + 1 # 下一个要训练的epoch
            # 加载最佳best_map (用于更新best_map, 不可或缺)
            best_map = checkpoint.get('best_map', 0.0)
            # 用于打印当前mAP@50~95
            current_map = checkpoint.get('current_map', 0.0)
            print(f"Checkpoint loaded successfully! "
                  f"Resuming from epoch {start_epoch}. "
                  f"Best mAP@50~95: {best_map:.4f}, "
                  f"Current mAP@50~95: {current_map:.4f}")
            print(f"Resume override the EMA params: "
                  f"\n    decay={ema.decay}, warmups={ema.warmups}, start={ema.start}, updates={ema.updates}")
            
            # 加载AFSS抗遗忘采样策略的状态, 兼容了老的pretrain权重(没有AFSS状态)
            if 'afss_state_dict' in checkpoint:
                afss_manager.state_dict = checkpoint['afss_state_dict']
                print("Loaded AFSS state successfully! Continuity maintained.")
                afss_manager.print_sufficiency_distribution()
            else:
                print("AFSS state not found in checkpoint, using default AFSS state.")
            
        else:
            print(f"ERROR: Checkpoint for resuming training: "
                  f"\n    {resume_path} is not found!")
            exit(1) # 如果指定了不存在的路径, 直接退出

    # for tuning training
    if tuning_path is not None:
        if os.path.isfile(tuning_path):
            print(f"Loading checkpoint for finetuning: \n   {tuning_path}...")
            checkpoint = torch.load(tuning_path, map_location=device)
            
            # 获取checkpoint中的num_classes
            checkpoint_num_classes = checkpoint.get('num_classes', train_config['num_classes'])
            current_num_classes = train_config['num_classes']
            
            # 如果num_classes不匹配，需要特殊处理分类头
            if checkpoint_num_classes != current_num_classes:
                print(f"Adapting classification heads from {checkpoint_num_classes} to {current_num_classes} classes...")
                # 首先加载除分类头外的所有权重
                model_weights = checkpoint['model_state_dict'] # 加载主模型权重
                
                # 处理主模型权重
                # 分类头存储检查点分类头数据(主要是把key存储起来, 后续用key来删除旧的分类头)
                class_embed_weights = {} 
                for key, value in list(model_weights.items()):
                    if 'class_embed' in key:
                        class_embed_weights[key] = value    
                # 删除检查点中的分类头权重
                for key in class_embed_weights:
                    del model_weights[key]
                # 加载无分类头权重的主模型
                dino_model.load_state_dict(model_weights, strict=False)
                # 分类头已经由模型创建时的_init_weights()初始化过了, 需额外初始化
                print(f"Classification heads reinitialized for {current_num_classes} classes.")
            else:
                # num_classes相同时，直接加载所有权重
                dino_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Checkpoint loaded successfully! ")
        else:
            print(f"ERROR: Checkpoint for finetuning: \n   {tuning_path} is not found!")
            exit(1) # 如果指定了不存在的路径, 直接退出
    
    # 初始化CSV文件保存训练和验证指标
    csv_dir = os.path.join(os.path.dirname(paths['model_best_path']), 'training_logs')
    os.makedirs(csv_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(csv_dir, f'training_metrics_{timestamp}.csv')
    
    # 写入CSV表头
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 
            'train_total_loss', 'train_cls_loss', 'train_box_loss', 'train_loc_loss',
            'train_main_cls_loss', 'train_main_box_loss', 'train_main_loc_loss',
            'val_total_loss', 'val_cls_loss', 'val_box_loss', 'val_loc_loss',
            'val_main_cls_loss', 'val_main_box_loss', 'val_main_loc_loss',
            'map@50', 'map@75', 'map@50:95', 'precision@50', 'recall@50', 'f1_max@50',
            'f1_max_score@50'
        ])
    
    print(f"Training metrics will be saved to: {csv_path}")
    
    ## training start, 如果是resume训练, 则从start_epoch开始, 否则直接退出从0 epoch开始
    # 开始记录训练时间time recoder
    start_time = datetime.now()
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    # training loop
    for epoch in range(start_epoch, train_config['epochs']):
        # 在每个epoch开始时更新学习率
        scheduler.step(epoch)
        # 每次epoch都需要更新数据集的epoch
        train_dataset.set_epoch(epoch) 
        dino_model.train()
        
        # ==========================================
        # AFSS 动态子集逻辑
        # ==========================================
        if epoch + 1 < afss_warmup_epochs:
            # 预热期：全量训练
            current_subset = list(range(len(train_dataset.original_img_lst)))
            for idx in current_subset:
                afss_manager.state_dict[idx]['ep'] = epoch
        else:
            # AFSS 阶段：动态获取本轮子集
            current_subset = afss_manager.get_epoch_subset(epoch)
            
        train_dataset.set_subset(current_subset)
        print(f"AFSS: Epoch {epoch+1} is using {len(current_subset)} / {len(train_dataset.original_img_lst)} images.")

        # 重新实例化 DataLoader, 同时每次更新新的随机种子确保每个 Epoch 的 Shuffle 顺序既随机又可严格复现
        g.manual_seed(train_config['seed'] + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=train_config['num_workers'],
            pin_memory=train_config['pin_memory'],
            prefetch_factor=prefetch_factor,
            worker_init_fn=seed_worker,
            generator=g,
        )
        # ==========================================

        epoch_losses = {
            'total': 0.0, 'cls': 0.0, 'box': 0.0, 'loc': 0.0, 'grad_norm': 0.0,
            'main_cls': 0.0, 'main_box': 0.0, 'main_loc': 0.0
        }
        batch_num = 0

        # 打印当前学习率
        # 进一步需要优化的是学习率调度与数据增强调度策略
        current_lr_backbone = optimizer.param_groups[0]['lr']
        current_lr_linear = optimizer.param_groups[1]['lr']
        current_lr_main = optimizer.param_groups[2]['lr']
        print(f"Epoch {epoch+1}/{train_config['epochs']} - LR: " +
              f"Backbone={current_lr_backbone:.1e}, Linear_proj={current_lr_linear:.1e}, Main={current_lr_main:.1e}")

        # 重置峰值内存统计(仅GPU)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        # 在epoch循环内, tqdm初始化前先打印表头
        # 打印完美对齐的表头
        epoch_len = len(str(train_config['epochs']))
        header = (
                    f"{'Epoch':>{epoch_len*2+1}} "
                    f"{'GPU_mem':>8} "
                    f"{'cls_loss':>10} "
                    f"{'box_loss':>10} "
                    f"{'loc_loss':>11} "
                    f"{'Instances':>10} "
                    f"{'Size':>8}"
                )
        print(f"{header}")
        # 自定义格式的进度条, 用set_description覆盖
        t = tqdm(train_loader, 
                 unit='batch', 
                 leave=True,  
                 bar_format='{desc} | {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_idx, batch in enumerate(t):
            nested_tensor, real_id, real_norm_bboxes, _, _ = batch
            # 构建targets 用于去噪训练
            targets = {}
            targets['labels'] = [i.cuda() for i in real_id]
            targets['boxes'] = [i.cuda() for i in real_norm_bboxes]
            
            # 模型输出是cxcywh的yolo格式, l1计算使用cxcywh, giou计算使用xyxy格式
            all_cls, all_bbox, dn_outs, \
            dec_out_corners, dec_out_ref_initials, \
            traditional_logits, traditional_bboxes, \
            fine_grained_out = dino_model(nested_tensor.tensors.to(device), nested_tensor.mask.to(device), targets=targets)
            
            total_loss, loss_dict, main_cls_loss,\
            main_box_loss, main_loc_loss = compute_train_loss(all_cls, all_bbox, dn_outs, dec_out_corners, dec_out_ref_initials,
                                                 traditional_logits, traditional_bboxes, fine_grained_out,
                                                 real_id, real_norm_bboxes, device=device, alpha=alpha, gamma=gamma,
                                                 class_loss_methed=cls_loss_method)
            
            optimizer.zero_grad()
            total_loss.backward()
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(dino_model.parameters(), max_norm)
            else:
                grad_total_norm = get_total_grad_norm(dino_model.parameters(), max_norm)
            optimizer.step()
            
            # 在优化器更新后更新EMA
            ema.update(dino_model)

            # === 构造 YOLO 风格描述行 ===
            # 1. 实例总数(当前 batch 中所有图像的目标数)
            instances = sum(len(ids) for ids in real_id)

            # 2. 图像尺寸(取第一张图的 HxW)
            img_h, img_w = nested_tensor.tensors.shape[-2:]

            # 3. GPU 显存(峰值, 单位 GB)
            if device.type == 'cuda':
                gpu_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                gpu_mem_str = f"{gpu_mem_gb:.2f}G"
            else:
                gpu_mem_str = "0.00G"

            # 4. 构造对齐的描述字符串（YOLO 风格）
            desc = (
                    f"{epoch+1:>{epoch_len}}/{train_config['epochs']} "
                    f"{gpu_mem_str:>8} "
                    f"{loss_dict.get('cls_loss', 0):>10.4f} "
                    f"{loss_dict.get('box_loss', 0):>10.4f} "
                    f"{loss_dict.get('loc_loss', 0):>11.4f} "
                    f"{instances:>10} "
                    f"{img_h:4d}x{img_w:<4d}"  # 使用x代替:更符合视觉
                )

            # 设置 tqdm 左侧描述
            t.set_description(desc)
            
            batch_num += 1
            epoch_losses['total'] += total_loss.item()
            epoch_losses['grad_norm'] += grad_total_norm.item()
            for k in ['cls', 'box', 'loc']:
                epoch_losses[k] += loss_dict[f'{k}_loss']
            epoch_losses['main_cls'] += main_cls_loss.item()
            epoch_losses['main_box'] += main_box_loss.item()
            epoch_losses['main_loc'] += main_loc_loss.item()

        # statistic of training epoch print
        print(f"\nEpoch {epoch+1} Training summary:")
        
        # 计算训练损失的平均值
        avg_train_total_loss = epoch_losses['total'] / batch_num
        avg_train_cls_loss = epoch_losses['cls'] / batch_num
        avg_train_box_loss = epoch_losses['box'] / batch_num
        avg_train_loc_loss = epoch_losses['loc'] / batch_num
        avg_grad_norm = epoch_losses['grad_norm'] / batch_num
        avg_train_main_cls_loss = epoch_losses['main_cls'] / batch_num
        avg_train_main_box_loss = epoch_losses['main_box'] / batch_num
        avg_train_main_loc_loss = epoch_losses['main_loc'] / batch_num
        # 打印训练损失
        print(f"  total_loss: {avg_train_total_loss:.4f}")
        print(f"  cls_loss: {avg_train_cls_loss:.4f}")
        print(f"  box_loss: {avg_train_box_loss:.4f}")
        print(f"  loc_loss: {avg_train_loc_loss:.4f}")
        print(f"  grad_norm: {avg_grad_norm:.4f}")
        print(f"  main_cls_loss: {avg_train_main_cls_loss:.4f}")
        print(f"  main_box_loss: {avg_train_main_box_loss:.4f}")
        print(f"  main_loc_loss: {avg_train_main_loc_loss:.4f}")

        # validate part
        # 根据配置选择验证模型: 使用EMA模型或原始模型
        if train_config['use_ema']:
            val_model = ema.module
        else:
            val_model = dino_model
        print(f"\nValidating EMA model: Epoch {epoch+1}...")
        val_metrics, all_p50, all_r50, all_score50, \
        current_map50, current_map75, current_map50t95, \
        val_total_loss, val_cls_loss, val_box_loss, val_loc_loss, \
        val_main_cls_loss, val_main_box_loss, val_main_loc_loss \
        = validate_model(val_imgpath=val_imgdir,
                         val_txtpath=val_txtdir,
                         seed_worker=seed_worker,
                         model=val_model,  # 使用ema模型 or 原始模型 进行验证
                         seed=train_config['seed'],
                         prefetch_factor=prefetch_factor,
                         num_classes=train_config['num_classes'],
                         num_queries=train_config['num_queries'],
                         batch_size=train_config['batch_size'],
                         workers=train_config['num_workers'],
                         gate_attn=train_config['gate_attention'],
                         pin_memory=train_config['pin_memory'],
                         scores_threshold=train_config['scores_threshold'],
                         alpha=alpha,
                         gamma=gamma,
                         cls_loss_method=cls_loss_method,
                         compute_loss=True,
                         max_size=train_config['max_size'],
                         val_size=train_config['val_size'],
                         use_ema=use_ema,
                         )
        
        printer_eval(val_metrics, all_p50, all_r50, all_score50, 
                   current_map50, current_map75, current_map50t95)
        # 打印验证损失
        if val_total_loss is not None:
            print(f"\nValidation Loss Summary:")
            print(f"  total_loss: {val_total_loss:.4f}")
            print(f"  cls_loss: {val_cls_loss:.4f}")
            print(f"  box_loss: {val_box_loss:.4f}")
            print(f"  loc_loss: {val_loc_loss:.4f}")
            print(f"  main_cls_loss: {val_main_cls_loss:.4f}")
            print(f"  main_box_loss: {val_main_box_loss:.4f}")
            print(f"  main_loc_loss: {val_main_loc_loss:.4f}")
        
        # 计算F1_max@IoU=0.50 and score@IoU=0.50
        if all_p50 + all_r50 == 0:
            f1_max50 = 0.0
        else:
            f1_max50 = 2 * all_p50 * all_r50 / (all_p50 + all_r50)
        
        # 保存当前epoch的训练和验证指标到CSV文件
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{avg_train_total_loss:.4f}",
                f"{avg_train_cls_loss:.4f}",
                f"{avg_train_box_loss:.4f}",
                f"{avg_train_loc_loss:.4f}",
                f"{avg_train_main_cls_loss:.4f}",
                f"{avg_train_main_box_loss:.4f}",
                f"{avg_train_main_loc_loss:.4f}",
                f"{val_total_loss:.4f}" if val_total_loss is not None else "N/A",
                f"{val_cls_loss:.4f}" if val_cls_loss is not None else "N/A",
                f"{val_box_loss:.4f}" if val_box_loss is not None else "N/A",
                f"{val_loc_loss:.4f}" if val_loc_loss is not None else "N/A",
                f"{val_main_cls_loss:.4f}" if val_main_cls_loss is not None else "N/A",
                f"{val_main_box_loss:.4f}" if val_main_box_loss is not None else "N/A",
                f"{val_main_loc_loss:.4f}" if val_main_loc_loss is not None else "N/A",
                f"{current_map50:.4f}",
                f"{current_map75:.4f}",
                f"{current_map50t95:.4f}",
                f"{all_p50:.4f}",
                f"{all_r50:.4f}",
                f"{f1_max50:.4f}",
                f"{all_score50:.4f}",
            ])
        
        # ==========================================
        # 将 AFSS 状态更新放在 Epoch 的末尾, 保存内容之前，使得训练更新的AFSS状态能够被保留在checkpoint中
        # ==========================================
        if epoch + 1 >= afss_warmup_epochs and (epoch + 1 - afss_warmup_epochs) % afss_epochs == 0:
            print(f"[AFSS State Update] Evaluating training set to update learning sufficiency...")
            # 拿到刚刚验证集算出来的最新最优阈值best_50_score
            # 加入一个底线保护, 防止早期模型极差时阈值过低
            current_conf_thresh = max(all_score50, 0.2)
            # 加载对应模型
            eval_model = ema.module if train_config['use_ema'] else dino_model
            
            # 内部是使用ValDatasets进行推理验证, 得到每一张图的P, R值, 作为真实推理时候的情况
            afss_manager.evaluate_and_update(
                model=eval_model,
                train_imgdir=train_imgdir,
                train_txtdir=train_txtdir,
                device=device,
                conf_thresh=current_conf_thresh, # 传入最精确的阈值
                max_size=train_config['max_size'],
                val_size=train_config['val_size'],
                num_workers=train_config['num_workers'],
                batch_size=train_config['batch_size'],
                pin_memory=train_config['pin_memory'],
                prefetch_factor=prefetch_factor,
                iou_thresh=0.5,
            )
            print(f"[AFSS State Update] Completed. Ready for Epoch {epoch+2}.\n")
        # ==========================================

        # best checkpoint save 
        current_map = current_map50t95 # model mAP@50~95
        if current_map > best_map:
            best_map = current_map
            best_checkpoint = paths['model_best_path']
            # 确保目录存在, os.makedirs() 会递归创建目录
            os.makedirs(os.path.dirname(best_checkpoint), exist_ok=True)  
            torch.save({
                # 保存当前完成的epoch
                'epoch': epoch, 
                # 保存原始模型权重, torch自带state_dict
                'model_state_dict': dino_model.state_dict(),
                # 保存ema模型权重与对应ema参数, ema自定义state_dict
                'ema_state_dict': ema.state_dict(),
                # 保存优化器状态, torch自带state_dict
                'optimizer_state_dict': optimizer.state_dict(), 
                # 保存调度器状态, 自定义制作state_dict, 保存配置初始学习率与当前训练轮次
                'scheduler_state_dict': scheduler.state_dict(), 
                # 保存 AFSS 状态, 自定义制作state_dict, 这是一个属性
                'afss_state_dict': afss_manager.state_dict,
                'best_map': best_map, # 保存最佳mAP@50~95(不可删除)
                'current_map': current_map, # 保存当前mAP@50~95
                'num_classes': train_config['num_classes'], # 保存类别数
            }, best_checkpoint)
            print(f"Save the best checkpoint to: {best_checkpoint}")
        
        # last checkpoint save
        last_checkpoint = paths['model_last_path']
        # 确保目录存在, os.makedirs() 会递归创建目录
        os.makedirs(os.path.dirname(last_checkpoint), exist_ok=True)
        torch.save({
                # 保存当前完成的epoch
                'epoch': epoch, 
                # 保存原始模型权重, torch自带state_dict
                'model_state_dict': dino_model.state_dict(),
                # 保存ema模型权重与对应ema参数, ema自定义state_dict
                'ema_state_dict': ema.state_dict(),
                # 保存优化器状态, torch自带state_dict  
                'optimizer_state_dict': optimizer.state_dict(), 
                # 保存调度器状态, 自定义制作state_dict, 保存配置初始学习率与当前训练轮次
                'scheduler_state_dict': scheduler.state_dict(), 
                # 保存 AFSS 状态, 自定义制作state_dict, 这是一个属性
                'afss_state_dict': afss_manager.state_dict,
                'best_map': best_map, # 保存最佳mAP@50~95(不可删除)
                'current_map': current_map, # 保存当前mAP@50~95
                'num_classes': train_config['num_classes'], # 保存类别数
                }, last_checkpoint)
        print(f"Save the last checkpoint to: {last_checkpoint}\n")

    # summary of the best model on validation
    print(f"Training Completed! Best model mAP@50~95: {best_map:.4f}")
    print("\nValidation result of the best model:")
    best_val_metrics, best_all_p50, best_all_r50, best_all_score50, \
    best_map50, best_map75, best_map50t95, \
    _, _, _, _, _, _, _ = validate_model(
        val_imgpath=val_imgdir, 
        val_txtpath=val_txtdir,
        seed_worker=seed_worker,
        prefetch_factor=prefetch_factor,
        model_path=paths['model_best_path'],
        seed=train_config['seed'],
        num_classes=train_config['num_classes'],
        num_queries=train_config['num_queries'],
        batch_size=train_config['batch_size'],
        workers=train_config['num_workers'],
        gate_attn=train_config['gate_attention'],
        pin_memory=train_config['pin_memory'],
        scores_threshold=train_config['scores_threshold'],
        compute_loss=False,
        max_size=train_config['max_size'],
        val_size=train_config['val_size'],
        use_ema=use_ema,
    )

    
    # print the best model validation
    printer_eval(best_val_metrics, best_all_p50, best_all_r50, best_all_score50, 
                    best_map50, best_map75, best_map50t95)
    
    # ====== 记录训练结束时间并打印总耗时 ======
    end_time = datetime.now()
    total_time = end_time - start_time
    print(f"\n{'='*60}")
    print(f"Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {total_time}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    # 创建解析器并添加参数
    args = parse_args()
    # 传入、覆盖、替换参数
    config = load_config(args.config, args)
    main(config, args.device)