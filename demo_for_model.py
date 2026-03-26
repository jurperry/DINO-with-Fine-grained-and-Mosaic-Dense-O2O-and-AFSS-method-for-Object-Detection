from datasets import TrainDataset_for_DETR, ValDataset_for_DETR, PredictDataset_for_DETR
from utils import collate_fn
from torch.utils.data import DataLoader
from models import GatedAttention_FineGrained_DINO_Swin

if __name__ == '__main__':
    imgdir_path = "your/image/path"
    txtdir_path = "your/label/path"
    batch_size = 2
    
    mydataset = TrainDataset_for_DETR(imgdir_path, txtdir_path, image_set="train")
    mydataset.set_epoch(0)
    
    mydataloader = DataLoader(mydataset, 
                                    batch_size=batch_size, 
                                    shuffle=True,
                                    collate_fn=collate_fn, 
                                    num_workers=0, 
                                    pin_memory=True)
    # model
    model = GatedAttention_FineGrained_DINO_Swin(num_queries=300, num_classes=80, gate_attn=True).cuda()
    model.eval()
    # inference
    for batch in mydataloader:
        nested_tensor, real_id, real_norm_bboxes, _, _ = batch
        imgs, masks, real_ids, real_bboxes = nested_tensor.tensors, nested_tensor.mask, real_id, real_norm_bboxes
        # 构建targets 用于去噪训练
        targets = {}
        targets['labels'] = [i.cuda() for i in real_ids]
        targets['boxes'] = [i.cuda() for i in real_bboxes]
        # forward
        all_cls, all_bbox, dn_outs, _, _, _, _, fine_grained = model(imgs.cuda(), mask=masks.cuda(), targets=targets)
        print(f"Number of decoder layers: {len(all_cls) - 2 }")
        print(f"Shape of first layer class predictions: {all_cls[0].shape}")
        print(f"Shape of first layer bbox predictions: {all_bbox[0].shape}")
        print(dn_outs['num_dn_groups'])
        if model.training:
            print(dn_outs['dn_traditional_logits'].shape)
            print(dn_outs['dn_dec_corners'].shape)
        print(fine_grained['reg_scale'])
        break
