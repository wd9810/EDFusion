import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import gammaln
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def calculate_reg_uncertainty(v, alpha, beta):
    alpha_minus_one = torch.clamp(alpha - 1, min=1e-3)
    
    epistemic = beta / (v * alpha_minus_one + 1e-8)
    aleatoric = beta / (alpha_minus_one + 1e-8)
    
    return epistemic + aleatoric, epistemic, aleatoric


def calculate_evidence_weight(nu_ir, alpha_ir, nu_vis, alpha_vis):
    evidence_input_ir = torch.clamp(1 + 2 * nu_ir + alpha_ir, max=1e6)
    evidence_input_vis = torch.clamp(1 + 2 * nu_vis + alpha_vis, max=1e6)
    evidence_reg_ir = torch.log(evidence_input_ir)
    evidence_reg_vis = torch.log(evidence_input_vis)
    
    reg_weight_ir = evidence_reg_ir / (evidence_reg_ir + evidence_reg_vis + 1e-8)
    reg_weight_vis = 1 - reg_weight_ir
    
    reg_weight_ir = torch.clamp(reg_weight_ir, 1e-6, 1 - 1e-6)
    reg_weight_vis = torch.clamp(reg_weight_vis, 1e-6, 1 - 1e-6)
    return reg_weight_ir, reg_weight_vis


def negative_log_likelihood_loss(v, alpha, beta, target, gamma):
    """
    负对数似然损失
    """
    eps = 1e-3
    v = torch.clamp(v, min=eps)  # 确保 v >= 1e-8 > 0
    alpha = torch.clamp(alpha, min=1.0 + eps)  # 确保 alpha > 1
    beta = torch.clamp(beta, min=eps)  # 确保 beta >= 1e-8 > 0
    
    Omega = 2 * beta * (1 + v)
    Omega = torch.clamp(Omega, min=eps)  # 确保 Ω > 0
    
    # 计算各项
    residual = target - gamma
    inner_term = (residual ** 2) * v + Omega
    
    nll_loss = 0.5 * torch.log(torch.pi / v) - alpha * torch.log(Omega) + \
               (alpha + 0.5) * torch.log(inner_term) + \
               gammaln(alpha) - gammaln(alpha + 0.5)
    
    return nll_loss


def evidence_regularization_loss(v, alpha, target, gamma):
    """
    证据正则化损失函数
    公式: L_i^R(w) = |y_i - E[μ_i]| ⋅ Φ = |y_i - γ| ⋅ (2v + α)
    其中总证据: Φ = 2v + α
    """
    
    # 计算绝对误差 |y_i - γ|
    absolute_error = torch.abs(target - gamma)
    
    # 计算总证据 Φ = 2v + α
    total_evidence = 2 * v + alpha
    
    # 正则化损失 = 绝对误差 × 总证据
    reg_loss = absolute_error * total_evidence
    
    return reg_loss.mean()


class TV_Loss(nn.Module):
    """Total Variation Loss"""
    
    def __init__(self, weight_vis=0.01, weight_ir=0.01):
        super(TV_Loss, self).__init__()
        self.weight_vis = weight_vis
        self.weight_ir = weight_ir
    
    def forward(self, vis_images, ir_images, fusion_images):
        """
        Args:
            vis_images: images [B, C, H, W]
            ir_images: images [B, C, H, W]
            fusion_images: Fused images [B, C, H, W]
        Returns:
            tv_loss: Total variation loss
        """
        tv_loss = 0
        
        # Calculate TV loss for visible light branch
        H1, W1 = vis_images.shape[2], vis_images.shape[3]
        R1 = vis_images - fusion_images
        L_vis = torch.pow(R1[:, :, 1:H1, :] - R1[:, :, 0:H1 - 1, :], 2).sum() + \
              torch.pow(R1[:, :, :, 1:W1] - R1[:, :, :, 0:W1 - 1], 2).sum()
        
        # Calculate TV loss for infrared branch
        H2, W2 = ir_images.shape[2], ir_images.shape[3]
        R2 = ir_images - fusion_images
        L_ir = torch.pow(R2[:, :, 1:H2, :] - R2[:, :, 0:H2 - 1, :], 2).sum() + \
              torch.pow(R2[:, :, :, 1:W2] - R2[:, :, :, 0:W2 - 1], 2).sum()
        
        tv_loss = self.weight_vis * L_vis + self.weight_ir * L_ir
        
        return tv_loss


class VIF_SSIM_Loss(nn.Module):
    """Visual Information Fidelity based SSIM Loss"""
    
    def __init__(self, kernel_size=11, num_channels=1, C=9e-4, device='cuda:0'):
        super(VIF_SSIM_Loss, self).__init__()
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.device = device
        self.C = C
        
        # Create average pooling kernel
        self.avg_kernel = torch.ones(num_channels, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        self.avg_kernel = self.avg_kernel.to(device)
    
    def forward(self, vis_images, ir_images, fusion_images):
        """
        Args:
            vis_images: Visible light images [B, C, H, W]
            ir_images: Infrared images [B, C, H, W]
            fusion_images: Fused images [B, C, H, W]
        Returns:
            ssim_loss: VIF-SSIM loss
        """
        batch_size, num_channels = vis_images.shape[0], vis_images.shape[1]
        
        # Calculate means and variances for visible images
        vis_images_mean = F.conv2d(vis_images, self.avg_kernel, stride=self.kernel_size, groups=num_channels)
        vis_images_var = torch.abs(F.conv2d(vis_images ** 2, self.avg_kernel, stride=self.kernel_size,
                                          groups=num_channels) - vis_images_mean ** 2)
        
        # Calculate means and variances for infrared images
        ir_images_mean = F.conv2d(ir_images, self.avg_kernel, stride=self.kernel_size, groups=num_channels)
        ir_images_var = torch.abs(F.conv2d(ir_images ** 2, self.avg_kernel, stride=self.kernel_size,
                                          groups=num_channels) - ir_images_mean ** 2)
        
        # Calculate means and variances for fusion images
        fusion_images_mean = F.conv2d(fusion_images, self.avg_kernel, stride=self.kernel_size, groups=num_channels)
        fusion_images_var = torch.abs(F.conv2d(fusion_images ** 2, self.avg_kernel, stride=self.kernel_size,
                                               groups=num_channels) - fusion_images_mean ** 2)
        
        # Calculate covariances
        vis_fusion_cov = F.conv2d(vis_images * fusion_images, self.avg_kernel, stride=self.kernel_size,
                                groups=num_channels) - vis_images_mean * fusion_images_mean
        ir_fusion_cov = F.conv2d(ir_images * fusion_images, self.avg_kernel, stride=self.kernel_size,
                                groups=num_channels) - ir_images_mean * fusion_images_mean
        
        C = torch.ones_like(fusion_images_mean) * self.C
        
        # Calculate luminance comparison
        ssim_l_vis_fusion = (2 * vis_images_mean * fusion_images_mean + C) / \
                          (vis_images_mean ** 2 + fusion_images_mean ** 2 + C)
        ssim_l_ir_fusion = (2 * ir_images_mean * fusion_images_mean + C) / \
                          (ir_images_mean ** 2 + fusion_images_mean ** 2 + C)
        
        # Calculate structure comparison
        ssim_s_vis_fusion = (vis_fusion_cov + C) / (vis_images_var + fusion_images_var + C)
        ssim_s_ir_fusion = (ir_fusion_cov + C) / (ir_images_var + fusion_images_var + C)
        
        # Adaptive weighting based on mean intensity
        score_vis_ir_fusion = (vis_images_mean > ir_images_mean) * ssim_l_vis_fusion * ssim_s_vis_fusion + \
                           (vis_images_mean <= ir_images_mean) * ssim_l_ir_fusion * ssim_s_ir_fusion
        
        ssim_loss = 1 - score_vis_ir_fusion.mean()
        
        return ssim_loss


def normalize_to_01(img):
    return (img + 1.0) / 2.0




def loss_tv(fused_img, ir_img, vis_img, evid_vis, evid_ir):
    wA = evid_ir.mean().detach()
    wB = evid_vis.mean().detach()
    tv_loss_fn = TV_Loss(weight_vis=wB, weight_ir=wA)
    tv_loss = tv_loss_fn(vis_img, ir_img, fused_img)
    return tv_loss

def loss_tv_no_evid(fused_img, ir_img, vis_img):
    tv_loss_fn_no_evid = TV_Loss()
    tv_loss_no_evid = tv_loss_fn_no_evid(vis_img, ir_img, fused_img)
    return tv_loss_no_evid


def loss_vif_ssim(fused_img, ir_img, vis_img):
    fused_01 = normalize_to_01(fused_img)
    ir_01 = normalize_to_01(ir_img)
    vis_01 = normalize_to_01(vis_img)
    vif_ssim_loss_fn = VIF_SSIM_Loss(kernel_size=11, num_channels=1,
                                     device='cuda' if torch.cuda.is_available() else 'cpu')  # 视觉保真度结构相似性损失
    ssim_loss = vif_ssim_loss_fn(vis_01, ir_01, fused_01)
    return ssim_loss


# --------------------- 预定义算子（避免重复创建） ---------------------
# 1. Scharr梯度算子（提前创建，复用）
scharr_x = torch.FloatTensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]])
scharr_x = scharr_x.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,3,3]
scharr_y = torch.FloatTensor([[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]])
scharr_y = scharr_y.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,3,3]


# 2. 高斯核（用于LoG和低频提取，提前创建）
def create_gaussian_kernel(kernel_size=7, sigma=1.0):
    """PyTorch实现高斯核"""
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = torch.from_numpy(kernel @ kernel.T).float().to(device)
    return kernel.unsqueeze(0).unsqueeze(0)  # [1,1,ks,ks]


gaussian_kernel = create_gaussian_kernel(7, 1.0)
laplacian_kernel = torch.FloatTensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).unsqueeze(0).unsqueeze(0).to(device)


def gradient_Sh(x):
    """Scharr梯度计算（PyTorch版，复用预定义核）"""
    # 适配多通道输入（如批量图像）
    if x.shape[1] > 1:
        scharr_x_exp = scharr_x.repeat(x.shape[1], 1, 1, 1)
        scharr_y_exp = scharr_y.repeat(x.shape[1], 1, 1, 1)
        d_x = F.conv2d(x, scharr_x_exp, stride=1, padding=1, groups=x.shape[1])
        d_y = F.conv2d(x, scharr_y_exp, stride=1, padding=1, groups=x.shape[1])
    else:
        d_x = F.conv2d(x, scharr_x, stride=1, padding=1)
        d_y = F.conv2d(x, scharr_y, stride=1, padding=1)
    d = torch.sqrt(torch.square(d_x) + torch.square(d_y) + 1e-8)  # 避免sqrt(0)
    return d


def LoScharr(x):
    """实现LoScharr（高斯模糊+拉普拉斯）"""
    # 步骤1：高斯模糊
    padding = (gaussian_kernel.shape[-1] - 1) // 2
    x_gau = F.conv2d(x, gaussian_kernel, padding=padding, groups=x.shape[1])
    # 步骤2：拉普拉斯
    x_gau_Scharr = gradient_Sh(x_gau)
    return x_gau_Scharr


def H_LoSchar(x):
    x_high_enhance = x + 0.5 * LoScharr(x)  # 降低增益到0.5，减少噪声
    return torch.clamp(x_high_enhance, -1.0, 1.0)


def enhance_recon_loss_func(evid_ir, evid_vis, fused, ir_img, vis_img):
    """
    :param evid_ir: 红外证据权重 [B,1,H,W]（像素级）
    :param evid_vis: 可见光证据权重 [B,1,H,W]（像素级）
    :param fused: 融合结果 [B,1,H,W]
    :param ir_img: 红外图像 [B,1,H,W]
    :param vis_img: 可见光图像 [B,1,H,W]
    :return: 重构损失值
    """

    wA = torch.clamp(evid_ir, 1e-6, 1.0)
    wB = torch.clamp(evid_vis, 1e-6, 1.0)
    

    grad_fused = H_LoSchar(fused)
    grad_ir = H_LoSchar(ir_img)
    grad_vis = H_LoSchar(vis_img)

    grad_loss = (wA * F.l1_loss(grad_fused, grad_ir, reduction='none') +
                 wB * F.l1_loss(grad_fused, grad_vis, reduction='none'))
    recon_loss = grad_loss.mean()
    
    return recon_loss


