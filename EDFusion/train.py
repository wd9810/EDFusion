import os
import time
import argparse
import h5py
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from utils import *
from model import FusionModel
import numpy as np
import torch


class PairedImageDataset(Dataset):
    """
    配对图像数据集
    """
    
    def __init__(self, ir_dir, vi_dir, patch_size=64, stride=32, transform=None):
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        
        # 获取红外和可见光图片文件
        self.ir_files = []
        self.vi_files = []
        
        # 获取所有红外图片
        for f in os.listdir(ir_dir):
            if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                self.ir_files.append(f)
        
        # 获取所有可见光图片
        for f in os.listdir(vi_dir):
            if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                self.vi_files.append(f)
        
        # 按文件名排序
        self.ir_files.sort()
        self.vi_files.sort()
        
        print(f"红外图片: {len(self.ir_files)} 张")
        print(f"可见光图片: {len(self.vi_files)} 张")
        
        # 完整路径
        self.ir_paths = [os.path.join(ir_dir, f) for f in self.ir_files]
        self.vi_paths = [os.path.join(vi_dir, f) for f in self.vi_files]
        
        # 检查配对
        if len(self.ir_paths) != len(self.vi_paths):
            print(f"⚠️ 警告: 红外({len(self.ir_paths)})和可见光({len(self.vi_paths)})图片数量不匹配")
            min_len = min(len(self.ir_paths), len(self.vi_paths))
            self.ir_paths = self.ir_paths[:min_len]
            self.vi_paths = self.vi_paths[:min_len]
            print(f"将使用前 {min_len} 对图片进行训练")
        
        # 预计算每个图片的patch数量
        self.patch_counts = []
        self.patch_indices = []  # (img_idx, patch_idx)映射
        
        for i in range(len(self.ir_paths)):
            ir_img = Image.open(self.ir_paths[i])
            w, h = ir_img.size
            
            if w < patch_size or h < patch_size:
                continue
            
            patches_per_img = 0
            for x in range(0, h - patch_size + 1, stride):
                for y in range(0, w - patch_size + 1, stride):
                    patches_per_img += 1
                    self.patch_indices.append((i, len(self.patch_counts)))
            
            self.patch_counts.append(patches_per_img)
        
        self.total_patches = len(self.patch_indices)
        print(f"总patch数: {self.total_patches}")
    
    def __len__(self):
        return self.total_patches
    
    def __getitem__(self, idx):
        img_idx, patch_idx = self.patch_indices[idx]
        
        # 加载红外图片
        ir_img = Image.open(self.ir_paths[img_idx])
        # 统一转换为单通道
        if ir_img.mode != 'L':
            ir_img = ir_img.convert('L')
        ir_array = np.array(ir_img, dtype=np.float32)
        
        # 加载可见光图片
        vi_img = Image.open(self.vi_paths[img_idx])
        # 统一转换为单通道
        if vi_img.mode != 'L':
            vi_img = vi_img.convert('L')
        vi_array = np.array(vi_img, dtype=np.float32)
        
        # 归一化到[-1, 1]
        ir_array = (ir_array - 127.5) / 127.5
        vi_array = (vi_array - 127.5) / 127.5
        
        # 计算patch位置
        h, w = ir_array.shape
        stride = self.stride
        patch_size = self.patch_size
        
        # 计算patch位置
        row = (patch_idx * stride) // (w - patch_size + 1)
        col = (patch_idx * stride) % (w - patch_size + 1)
        x = min(row * stride, h - patch_size)
        y = min(col * stride, w - patch_size)
        
        # 提取patch
        ir_patch = ir_array[x:x + patch_size, y:y + patch_size]
        vi_patch = vi_array[x:x + patch_size, y:y + patch_size]
        
        # 添加通道维度 [H,W] -> [H,W,1]
        ir_patch = ir_patch[..., np.newaxis]
        vi_patch = vi_patch[..., np.newaxis]
        
        # 转换为Tensor [C,H,W]
        ir_tensor = torch.from_numpy(ir_patch).permute(2, 0, 1).float()
        vi_tensor = torch.from_numpy(vi_patch).permute(2, 0, 1).float()
        
        return ir_tensor, vi_tensor


def prepare_patches_improved(image_dir, output_h5_path, patch_size=64, stride=32):
    """
    改进的预处理函数，处理彩色/灰度混合图片
    """
    image_files = sorted([f for f in os.listdir(image_dir)
                          if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    patches = []
    
    print(f"处理目录: {image_dir}, 图片数: {len(image_files)}")
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path)
        
        # 统一转换为灰度图
        if img.mode != 'L':
            img = img.convert('L')
        
        img_array = np.array(img, dtype=np.float32)
        
        # 归一化到[-1, 1]
        img_array = (img_array - 127.5) / 127.5
        
        h, w = img_array.shape
        
        # 生成patch
        patch_count = 0
        for x in range(0, h - patch_size + 1, stride):
            for y in range(0, w - patch_size + 1, stride):
                patch = img_array[x:x + patch_size, y:y + patch_size]
                patches.append(patch[..., np.newaxis])  # 增加通道维度
                patch_count += 1
        
        if patch_count == 0:
            print(f"⚠️ 图片 {img_file} 尺寸太小 ({w}x{h})，无法生成 {patch_size}x{patch_size} 的patch")
    
    patches = np.array(patches)
    print(f"生成的patch总数: {patches.shape[0]}")
    
    # 保存为h5文件
    with h5py.File(output_h5_path, 'w') as hf:
        hf.create_dataset('data', data=patches)
        hf.attrs['patch_size'] = patch_size
        hf.attrs['stride'] = stride
        hf.attrs['num_images'] = len(image_files)
    
    return patches.shape[0]


def train(model, device, train_loader, optimizer, epoch, writer):
    model.train()
    total_loss = 0
    
    for batch_idx, (ir_img, vis_img) in enumerate(train_loader):
        ir_img, vis_img = ir_img.to(device), vis_img.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(ir_img, vis_img)
        
        # 提取需要的输出
        base_fused = outputs['fused']  # 像素级融合结果 [B, 1, H, W]
        mu_f = outputs["mu_f"]
        
        # 回归参数（用于回归损失）
        gamma_ir, nu_ir, alpha_ir, beta_ir = outputs['ir_params']  # 图像A的NIG参数
        gamma_vis, nu_vis, alpha_vis, beta_vis = outputs['vis_params']  # 图像B的NIG参数
        
        # 计算证据强度
        evid_ir, evid_vis = calculate_evidence_weight(nu_ir, alpha_ir, nu_vis, alpha_vis)
        
        # 计算各项损失
        # 1、回归损失
        nll_loss_ir = negative_log_likelihood_loss(nu_ir, alpha_ir, beta_ir, ir_img, gamma_ir)
        nll_loss_vis = negative_log_likelihood_loss(nu_vis, alpha_vis, beta_vis, vis_img, gamma_vis)
        nll_loss = (nll_loss_ir.mean() + nll_loss_vis.mean())
        
        reg_reg_loss_ir = evidence_regularization_loss(nu_ir, alpha_ir, ir_img, gamma_ir)
        reg_reg_loss_vis = evidence_regularization_loss(nu_vis, alpha_vis, vis_img, gamma_vis)
        reg_reg_loss = (reg_reg_loss_ir + reg_reg_loss_vis).mean()
        
        # 3. 加权重构损失
        recon_loss = enhance_recon_loss_func(evid_ir, evid_vis, base_fused, ir_img, vis_img)

        # 4.TV loss
        tv_loss = loss_tv(base_fused, ir_img, vis_img, evid_vis, evid_ir)
        
        # 5.SSIM loss
        ssim_loss = loss_vif_ssim(base_fused, ir_img, vis_img)
        
        # 总损失
        loss = (0.5 * nll_loss + 0.2 * reg_reg_loss + 5 * recon_loss + 0.01 * tv_loss + ssim_loss)
        
        # 反向传播
        loss.backward()
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.6f} '
                  f'NLL: {nll_loss.item():.6f} '
                  f'Reg: {reg_reg_loss.item():.6f} '
                  f'TV: {tv_loss.item():.6f} '
                  f'SSIM: {ssim_loss.item():.6f} '
                  f'Recon: {recon_loss.item():.6f}')
            
            # 记录到TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train_total', loss.item(), global_step)
            writer.add_scalar('Loss/NLL', nll_loss.item(), global_step)
            writer.add_scalar('Loss/Reg_reg', reg_reg_loss.item(), global_step)
            writer.add_scalar('Loss/TV', tv_loss.item(), global_step)
            writer.add_scalar('Loss/SSIM', ssim_loss.item(), global_step)
            writer.add_scalar('Loss/recon', recon_loss.item(), global_step)
    
    avg_loss = total_loss / len(train_loader)
    print(f'\nTrain Epoch: {epoch} Average loss: {avg_loss:.6f}')
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Image Fusion with Deep Evidential Learning')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--ir-dir', type=str, default='./data/train/ir/',
                        help='Directory containing infrared images')
    parser.add_argument('--vi-dir', type=str, default='./data/train/vi/',
                        help='Directory containing visible images')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Directory to save outputs')
    parser.add_argument('--patch-size', type=int, default=64,
                        help='Size of image patches (default: 64)')
    parser.add_argument('--stride', type=int, default=32,
                        help='Stride for patch extraction (default: 32)')
    parser.add_argument('--use-h5', action='store_true', default=True,
                        help='Use HDF5 cache (faster but requires disk space)')
    args = parser.parse_args()
    
    # 设备设置
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print(f"使用设备: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'patches'), exist_ok=True)
    
    # 数据加载
    if args.use_h5:
        # 使用HDF5缓存
        ir_h5_path = os.path.join(args.output_dir, 'patches', 'ir_patches.h5')
        vis_h5_path = os.path.join(args.output_dir, 'patches', 'vi_patches.h5')
        
        if not os.path.exists(ir_h5_path):
            print("准备红外图片patches...")
            num_ir_patches = prepare_patches_improved(args.ir_dir, ir_h5_path, args.patch_size, args.stride)
            print(f"红外patches数量: {num_ir_patches}")
        
        if not os.path.exists(vis_h5_path):
            print("准备可见光图片patches...")
            num_vi_patches = prepare_patches_improved(args.vi_dir, vis_h5_path, args.patch_size, args.stride)
            print(f"可见光patches数量: {num_vi_patches}")
        
        # 检查patch数量是否匹配
        with h5py.File(ir_h5_path, 'r') as hf:
            ir_patches = hf['data'][:]
        with h5py.File(vis_h5_path, 'r') as hf:
            vis_patches = hf['data'][:]
        
        if len(ir_patches) != len(vis_patches):
            print(f"⚠️ 红外和可见光patch数量不匹配: IR={len(ir_patches)}, VI={len(vis_patches)}")
            min_len = min(len(ir_patches), len(vis_patches))
            ir_patches = ir_patches[:min_len]
            vis_patches = vis_patches[:min_len]
            print(f"将使用前 {min_len} 个patches")
        
        # 使用自定义Dataset
        class H5PatchDataset(Dataset):
            def __init__(self, ir_patches, vi_patches):
                self.ir_patches = ir_patches
                self.vi_patches = vi_patches
            
            def __len__(self):
                return len(self.ir_patches)
            
            def __getitem__(self, idx):
                ir_patch = torch.from_numpy(self.ir_patches[idx]).permute(2, 0, 1).float()
                vi_patch = torch.from_numpy(self.vi_patches[idx]).permute(2, 0, 1).float()
                return ir_patch, vi_patch
        
        train_dataset = H5PatchDataset(ir_patches, vis_patches)
    
    else:
        # 使用直接加载模式
        print("使用直接图片加载模式...")
        train_dataset = PairedImageDataset(
            ir_dir=args.ir_dir,
            vi_dir=args.vi_dir,
            patch_size=args.patch_size,
            stride=args.stride
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows下建议设为0
        pin_memory=True
    )
    
    print(f"数据加载完成，总batch数: {len(train_loader)}")
    
    # 模型初始化
    model = FusionModel().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 学习率调度器
    scheduler = StepLR(optimizer, step_size=20, gamma=0.7)
    
    # TensorBoard记录
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    def create_checkpoint(model, optimizer, scheduler, epoch, avg_loss, best_loss, args):
        """创建检查点的辅助函数"""
        return {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'current_loss': avg_loss,
            'training_args': vars(args),
            'timestamp': time.time(),
        }
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # 训练
        avg_loss = train(model, device, train_loader, optimizer, epoch, writer)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        # 更新最佳损失
        is_best = False
        if avg_loss < best_loss:
            best_loss = avg_loss
            is_best = True
        
        # 保存最佳模型
        if is_best:
            checkpoint = create_checkpoint(model, optimizer, scheduler, epoch, avg_loss, best_loss, args)
            torch.save(checkpoint, os.path.join(args.output_dir, 'models', 'best_model.pth'))
            print(f'🎯 新最佳模型在epoch {epoch}, loss: {best_loss:.6f}')
        
        # 定期保存检查点
        if epoch % 10 == 0 or epoch == args.epochs:
            checkpoint = create_checkpoint(model, optimizer, scheduler, epoch, avg_loss, best_loss, args)
            torch.save(checkpoint, os.path.join(args.output_dir, 'models', f'checkpoint_epoch_{epoch}.pth'))
            print(f'💾 检查点保存在epoch {epoch}')
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Loss/best', best_loss, epoch)
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch:03d}/{args.epochs} | '
              f'时间: {epoch_time:.1f}s | '
              f'Loss: {avg_loss:.6f} | '
              f'最佳: {best_loss:.6f} | '
              f'学习率: {current_lr:.6f}')
    
    # 保存最终模型
    final_checkpoint = create_checkpoint(model, optimizer, scheduler, args.epochs, avg_loss, best_loss, args)
    torch.save(final_checkpoint, os.path.join(args.output_dir, 'models', 'final_model.pth'))
    
    print(f'✅ 训练完成! 最佳loss: {best_loss:.6f}')
    writer.close()


if __name__ == '__main__':
    main()