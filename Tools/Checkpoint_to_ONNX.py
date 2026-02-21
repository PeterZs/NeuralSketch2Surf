import torch
import os
from train112TVloss import SwinReconstructionModule, InferenceModel 

def export_to_onnx():
    # path
    ckpt_path = "/home/ids/asureshk-22/Hongsheng/checkpoint/checkpoint112_TVloss/ckpt/epoch=43-val/mean_dice=0.8732.ckpt"  
    save_path = "/home/ids/asureshk-22/Hongsheng/checkpoint/model_universal.onnx" # Note that the suffix is .onnx.
    img_size = 112
    
    print(f"Loading from {ckpt_path}...")
    model_pl = SwinReconstructionModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    
    deploy_model = InferenceModel(model_pl.backbone, model_pl.refiner)
    deploy_model.eval()
    
    # (Batch_size=1, Channel=1, D, H, W)
    dummy_input = torch.randn(1, 1, img_size, img_size, img_size)
    
    # output ONNX
    print("Exporting to ONNX...")
    
    dynamic_axes = {
        'input': {0: 'batch_size'}, 
        'output': {0: 'batch_size'}
    }

    torch.onnx.export(
        deploy_model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    
    print(f"✅ ONNX model saved to {save_path}")
    print("This model can now run on BOTH GPU and CPU without modification!")

if __name__ == "__main__":
    export_to_onnx()