import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import random
import glob
import numpy as np
import torch
import torch.nn as nn

try:
    torch.serialization.add_safe_globals([argparse.Namespace])
except AttributeError:
    pass

# -A100 Acceleration
torch.set_float32_matmul_precision('medium')

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from monai.losses import DiceLoss
from monai.metrics import DiceMetric, MeanIoU


from Network.SwinUNETRV2.swin_unetr import SwinUNETR
from Network.refinement_net import RefinementNet


# TV Loss & DiceWCE
class TotalVariationLoss(nn.Module):
    """For constraining surface smoothness"""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x, w_x, d_x = x.size()[2], x.size()[3], x.size()[4]
        
        count_h = self._tensor_size(x[:, :, 1:, :, :])
        count_w = self._tensor_size(x[:, :, :, 1:, :])
        count_d = self._tensor_size(x[:, :, :, :, 1:])
        
        h_tv = torch.pow((x[:, :, 1:, :, :] - x[:, :, :h_x - 1, :, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:, :] - x[:, :, :, :w_x - 1, :]), 2).sum()
        d_tv = torch.pow((x[:, :, :, :, 1:] - x[:, :, :, :, :d_x - 1]), 2).sum()
        
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w + d_tv / count_d) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3] * t.size()[4]

class DiceWCELoss(nn.Module):
    def __init__(self, wce_weight=1.0):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True, batch=True)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([wce_weight]))

    def forward(self, pred, target):
        if self.bce.pos_weight.device != pred.device:
            self.bce.pos_weight = self.bce.pos_weight.to(pred.device)
        return self.dice(pred, target) + self.bce(pred, target)


# Dataset Definition
class VoxelDataset(Dataset):
    def __init__(self, file_list, root_dir, img_size=112, transform=None):
        self.file_list = file_list
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        item = self.file_list[idx]
        input_path = os.path.join(self.root_dir, item['input'])
        label_path = os.path.join(self.root_dir, item['label'])

        try:
            input_data = np.load(input_path).astype(np.float32)
            label_data = np.load(label_path).astype(np.float32)
        except Exception as e:
            print(f"Error loading {input_path}: {e}")
            return torch.zeros((1, self.img_size, self.img_size, self.img_size)), \
                   torch.zeros((1, self.img_size, self.img_size, self.img_size))

        if input_data.ndim == 3: input_data = input_data[None, ...]
        if label_data.ndim == 3: label_data = label_data[None, ...]

        if self.transform:
            # 1. Rotation
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                input_data = np.rot90(input_data, k, axes=(1, 3)).copy()
                label_data = np.rot90(label_data, k, axes=(1, 3)).copy()
            # 2. Flip
            if random.random() > 0.5:
                input_data = np.flip(input_data, axis=3).copy()
                label_data = np.flip(label_data, axis=3).copy()
            # 3. Gaussian Noise
            if random.random() < 0.5:
                noise = np.random.normal(0, 0.02, input_data.shape).astype(np.float32)
                input_data = input_data + noise

        return torch.from_numpy(input_data), torch.from_numpy(label_data)

# Data Module Definition
class VoxelDataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.seed = args.seed
        self.img_size = args.img_size

    def setup(self, stage=None):
        """
        Prepare data:
        1. Find input and label files in subdirectories.
        2. Pair them based on filenames.
        3. Split dataset into Train (80%) and Val (20%) based on unique Object IDs 
        """
        geo_dir = os.path.join(self.data_dir, "voxelize_geo")
        label_dir = os.path.join(self.data_dir, "voxelize_label")
        if not os.path.exists(geo_dir): raise ValueError(f"Not found: {geo_dir}")
        
        all_files = glob.glob(os.path.join(geo_dir, "*.npy"))
        file_items = []
        for f_path in all_files:
            f_name = os.path.basename(f_path) 
            mid = f_name.split('_')[0]
            l_name = f_name.replace("voxelize_geodesic", "voxelize_label")
            if os.path.exists(os.path.join(label_dir, l_name)):
                file_items.append({'id': mid, 'input': os.path.join("voxelize_geo", f_name), 'label': os.path.join("voxelize_label", l_name)})
        
        print(f"Found {len(file_items)} pairs.")
        
        # Split by ID
        unique_ids = sorted(list(set([x['id'] for x in file_items])))
        random.seed(self.seed)
        random.shuffle(unique_ids)

        n_train = int(len(unique_ids) * 0.8)
        train_ids = set(unique_ids[:n_train])
        val_ids = set(unique_ids[n_train:])
        
        self.train_data = [x for x in file_items if x['id'] in train_ids]
        self.val_data = [x for x in file_items if x['id'] in val_ids]

    def train_dataloader(self):
        return DataLoader(VoxelDataset(self.train_data, self.data_dir, self.img_size, True), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(VoxelDataset(self.val_data, self.data_dir, self.img_size, False), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)


#Lightning Training Core
class SwinReconstructionModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        # Backbone
        self.backbone = SwinUNETR(
            img_size=(args.img_size, args.img_size, args.img_size),
            in_channels=1,
            out_channels=1,
            feature_size=args.feature_size,
            use_v2=True,
            drop_rate=args.dropout
        )
        
        # Refiner
        self.refiner = RefinementNet(in_channels=1, hidden_dim=32, num_blocks=3)
        
        # Losses
        self.loss_fn = DiceWCELoss(wce_weight=args.wce_weight)
        self.tv_loss = TotalVariationLoss(weight=args.tv_weight)
        
        # Metrics
        self.train_dice = DiceMetric(include_background=True, reduction="mean")
        self.val_dice = DiceMetric(include_background=True, reduction="mean")
        self.train_iou = MeanIoU(include_background=True, reduction="mean")
        self.val_iou = MeanIoU(include_background=True, reduction="mean")

    def forward(self, x):
        coarse_logits = self.backbone(x)
        fine_logits = self.refiner(coarse_logits)
        return fine_logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Forward pass
        coarse_logits = self.backbone(x)
        fine_logits = self.refiner(coarse_logits)
        
        # Compute Loss
        
        # Coarse stage Loss (weight 0.5): Focus Swin on overall shape
        loss_coarse = self.loss_fn(coarse_logits, y)
        
        # Refinement stage Loss (weight 1.0): Focus Refiner on final quality
        loss_fine = self.loss_fn(fine_logits, y)
        
        # TV Loss Only constrain final output, enforce smoothness
        probs_fine = torch.sigmoid(fine_logits)
        loss_tv = self.tv_loss(probs_fine)
        
        # Total Loss
        total_loss = 0.5 * loss_coarse + 1.0 * loss_fine + loss_tv
        
        with torch.no_grad():
            preds = (probs_fine > 0.5).float()
            self.train_dice(y_pred=preds, y=y)
            self.train_iou(y_pred=preds, y=y)

        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/loss_coarse", loss_coarse, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("train/loss_fine", loss_fine, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("train/tv_loss", loss_tv, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        return total_loss

    def on_train_epoch_end(self):
        mean_dice = self.train_dice.aggregate().item()
        mean_iou = self.train_iou.aggregate().item()
        self.train_dice.reset()
        self.train_iou.reset()
        self.log("train/mean_dice", mean_dice, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/mean_iou", mean_iou, prog_bar=False, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        fine_logits = self(x)
        loss = self.loss_fn(fine_logits, y)
        
        preds = (torch.sigmoid(fine_logits) > 0.5).float()
        self.val_dice(y_pred=preds, y=y)
        self.val_iou(y_pred=preds, y=y)
        
        self.log("val/loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        mean_dice = self.val_dice.aggregate().item()
        mean_iou = self.val_iou.aggregate().item()
        self.val_dice.reset()
        self.val_iou.reset()
        self.log("val/mean_dice", mean_dice, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/mean_iou", mean_iou, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.args.lr,
            weight_decay=0.05
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6),
            "monitor": "val/loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
    
# A pure PyTorch wrapper specifically designed for exporting and inference
class InferenceModel(nn.Module):
    def __init__(self, backbone, refiner):
        super().__init__()
        self.backbone = backbone
        self.refiner = refiner

    def forward(self, x):
        coarse_logits = self.backbone(x)
        fine_logits = self.refiner(coarse_logits)
        return torch.sigmoid(fine_logits)



# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset Directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Save Directory")
    parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of CPU Workers")
    
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--wce_weight", type=float, default=1.0) 
    parser.add_argument("--tv_weight", type=float, default=0.1, help="Total Variation Loss Weight")
    
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--feature_size", type=int, default=24)
    
    parser.add_argument("--project", type=str, default="3D_Sketch_112", help="Wandb Project")
    parser.add_argument("--name", type=str, default=None, help="Wandb Experiment Name")
    parser.add_argument("--seed", type=int, default=42) 
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, "ckpt"),
        filename="{epoch:02d}-{val/mean_dice:.4f}",
        save_top_k=3,
        monitor="val/mean_dice",
        mode="max",
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/mean_dice", min_delta=0.0005, patience=15, verbose=True, mode="max"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    wandb_logger = WandbLogger(project=args.project, name=args.name, save_dir=args.save_dir)

    dm = VoxelDataModule(args)
    model = SwinReconstructionModule(args)

    if args.gpus == -1: devices = torch.cuda.device_count()
    else: devices = args.gpus
    strategy = "ddp_find_unused_parameters_true" if devices > 1 else "auto"

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=devices, 
        strategy=strategy,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=10,
        sync_batchnorm=True if devices > 1 else False
    )

    ckpt_path = os.path.join(args.save_dir, "ckpt", "last.ckpt")
    if os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    else:
        print("Starting new training...")
        trainer.fit(model, datamodule=dm)

    if trainer.global_rank == 0:
        print("\n✨ Exporting best model to TorchScript...")
        best_ckpt = checkpoint_callback.best_model_path
        if best_ckpt:
            try:
                best_pl_model = SwinReconstructionModule.load_from_checkpoint(best_ckpt, args=args)
                deploy_model = InferenceModel(best_pl_model.backbone, best_pl_model.refiner)
                
                deploy_model.eval()
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                deploy_model.to(device)
                
                dummy_input = torch.randn(1, 1, args.img_size, args.img_size, args.img_size).to(device).float()
                
                print(f"Tracing model on device: {device} ...")
                
                traced_model = torch.jit.trace(deploy_model, dummy_input, check_trace=False)
                
                deploy_path = os.path.join(args.save_dir, "best_model_jit.pt")
                torch.jit.save(traced_model, deploy_path)
                print(f"Success! Deployment model saved to: {deploy_path}")
                
            except Exception as e:
                print(f"Export failed: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()