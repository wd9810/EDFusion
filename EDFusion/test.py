import os
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import FusionModel


def rgb_to_ycbcr_corrected(image_tensor):
    """RGB转YCbCr"""
    r = image_tensor[:, 0:1, :, :]
    g = image_tensor[:, 1:2, :, :]
    b = image_tensor[:, 2:3, :, :]
    
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    
    cb = cb + 0.5
    cr = cr + 0.5
    
    return torch.cat([y, cb, cr], dim=1)


def ycbcr_to_rgb_corrected(image_tensor):
    """YCbCr转RGB"""
    y = image_tensor[:, 0:1, :, :]
    cb = image_tensor[:, 1:2, :, :] - 0.5
    cr = image_tensor[:, 2:3, :, :] - 0.5
    
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    
    return torch.cat([r, g, b], dim=1).clamp(-1, 1)


def test(model, device, ir_dir, vis_dir, output_dir):
    """
    彩色图像融合测试函数
    """
    model.eval()
    
    # 只创建必要的输出目录
    os.makedirs(os.path.join(output_dir, 'fused_rgb'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'f_uncertainty_maps'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'fused_with_uncertainty'), exist_ok=True)
    
    def custom_normalize_gray(img):
        img_array = np.array(img, dtype=np.float32)
        return (img_array - 127.5) / 127.5
    
    def custom_normalize_rgb(img):
        img_array = np.array(img, dtype=np.float32)
        return (img_array - 127.5) / 127.5
    
    def tensor_to_uint8(tensor):
        """将张量转换为uint8图像"""
        img = tensor.detach().cpu().numpy()
        img = np.squeeze(img)
        
        if img.ndim == 3:
            if img.shape[0] in [1, 3]:
                img = img.transpose(1, 2, 0)
                if img.shape[-1] == 1:
                    img = np.squeeze(img)
        
        img = (img * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        return img
    
    # 获取图像文件列表
    ir_files = sorted([f for f in os.listdir(ir_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    vis_files = sorted(
        [f for f in os.listdir(vis_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    
    assert len(ir_files) == len(vis_files), f"图像数量不匹配: {len(ir_files)} vs {len(vis_files)}"
    
    print(f"找到 {len(ir_files)} 对图像")
    
    for i, (ir_file, vis_file) in enumerate(zip(ir_files, vis_files)):
        ir_path = os.path.join(ir_dir, ir_file)
        vis_path = os.path.join(vis_dir, vis_file)
        
        try:
            # 加载图像
            ir_img = Image.open(ir_path).convert('L')
            vis_img = Image.open(vis_path).convert('RGB')
            
            # 检查尺寸是否匹配
            if ir_img.size != vis_img.size:
                print(f"⚠️ 尺寸不匹配: {ir_file} ({ir_img.size}) 和 {vis_file} ({vis_img.size})，调整VIS图像尺寸...")
                vis_img = vis_img.resize(ir_img.size, Image.Resampling.LANCZOS)
            
            print(f"处理 {i + 1}/{len(ir_files)}: {ir_file}")
            
            # 转换为张量
            ir_array = custom_normalize_gray(ir_img)
            vis_array = custom_normalize_rgb(vis_img)
            
            ir_tensor = torch.from_numpy(ir_array).unsqueeze(0).unsqueeze(0).float().to(device)
            vis_tensor = torch.from_numpy(vis_array.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            
            # 转换为YCbCr颜色空间
            vis_ycbcr = rgb_to_ycbcr_corrected(vis_tensor)
            vis_y = vis_ycbcr[:, 0:1, :, :]  # 亮度通道
            vis_cbcr = vis_ycbcr[:, 1:3, :, :]  # 色度通道
            
            # 进行融合
            with torch.no_grad():
                outputs = model(ir_tensor, vis_y)
            
            # 获取融合结果
            fused_y = outputs['fused']
            f_uncertainty = outputs['U_f']  # 获取f_uncertainty
            
            # 重建彩色图像
            fused_ycbcr = torch.cat([fused_y, vis_cbcr], dim=1)
            fused_rgb = ycbcr_to_rgb_corrected(fused_ycbcr)
            
            # 转换为numpy数组
            fused_img = tensor_to_uint8(fused_rgb)
            f_uncertainty_np = f_uncertainty.squeeze().cpu().numpy()
            
            # 归一化f_uncertainty到[0,1]
            f_uncertainty_normalized = (f_uncertainty_np - f_uncertainty_np.min()) / (
                        f_uncertainty_np.max() - f_uncertainty_np.min() + 1e-8)
            
            # 保存结果
            base_name = os.path.splitext(ir_file)[0]
            
            # 1. 保存融合图像
            fused_img_pil = Image.fromarray(fused_img)
            fused_img_pil.save(os.path.join(output_dir, 'fused_rgb', f'{base_name}_fused.png'))
            
            # 2. 保存f_uncertainty热力图
            plt.figure(figsize=(10, 8))
            im = plt.imshow(f_uncertainty_np, cmap='viridis')
            plt.colorbar(im, label='F_Uncertainty')
            plt.axis('off')
            plt.title(f'F_Uncertainty Map: {base_name}')
            plt.savefig(os.path.join(output_dir, 'f_uncertainty_maps', f'{base_name}_f_uncertainty.png'),
                        bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
            
            # 3. 创建并保存叠加图像
            cmap = plt.get_cmap('viridis')
            uncertainty_colored = (cmap(f_uncertainty_normalized)[:, :, :3] * 255).astype(np.uint8)
            alpha = 0.5
            overlayed_img = (fused_img * (1 - alpha) + uncertainty_colored * alpha).astype(np.uint8)
            
            Image.fromarray(overlayed_img).save(
                os.path.join(output_dir, 'fused_with_uncertainty', f'{base_name}_fused_with_uncertainty.png'))
            
            print(f"✅ 保存: {base_name}_fused.png, {base_name}_f_uncertainty.png")
        
        except Exception as e:
            print(f"❌ 处理 {ir_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue


def load_model_with_checkpoint(model_path, device):
    """
    加载模型和检查点
    """
    model = FusionModel().to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '未知')
        best_loss = checkpoint.get('best_loss', '未知')
        print(f"✅ 加载检查点 - 轮次: {epoch}, 最佳损失: {best_loss:.6f}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ 直接加载模型参数")
    
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='彩色图像融合测试')
    
    parser.add_argument('--ir-dir', type=str, default='./data/test/ir/',
                        help='包含单通道IR图像的目录')
    parser.add_argument('--vis-dir', type=str, default='./data/test/vi/',
                        help='包含多通道VIS彩色图像的目录')
    parser.add_argument('--model-path', type=str, default='./output/models/final_model.pth',
                        help='训练好的模型检查点路径')
    parser.add_argument('--output-dir', type=str, default='./test_output/',
                        help='输出目录')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='禁用CUDA')
    
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 加载模型
        model = load_model_with_checkpoint(args.model_path, device)
        
        # 测试
        test(model, device, args.ir_dir, args.vis_dir, args.output_dir)
        
        print(f"\n🎉 融合测试完成!")
        print(f"   融合RGB图像保存到: {os.path.join(args.output_dir, 'fused_rgb')}")
        print(f"   F_Uncertainty图保存到: {os.path.join(args.output_dir, 'f_uncertainty_maps')}")
        print(f"   叠加图像保存到: {os.path.join(args.output_dir, 'fused_with_uncertainty')}")
    
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()