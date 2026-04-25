import torch.nn as nn
import torch.nn.functional as F


# ===============================
# 基础证据回归网络
# 输出: μ, λ, α, β
# ===============================
class EvidenceRegressionNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super(EvidenceRegressionNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(base_channels, 4, kernel_size=1)

    def forward(self, x):

        feat = self.encoder(x)
        out = self.out_conv(feat)

        mu = out[:, 0:1, :, :]

        # 保证参数合法
        nu_ = F.softplus(out[:, 1:2, :, :]) + 1e-6
        alpha = F.softplus(out[:, 2:3, :, :]) + 1.0 + 1e-6
        beta = F.softplus(out[:, 3:4, :, :]) + 1e-6

        return mu, nu_, alpha, beta


# ===============================
# NIG 概率融合模块
# ===============================
class NIGFusion(nn.Module):
    def __init__(self, gamma=0.5):
        super(NIGFusion, self).__init__()
        self.gamma = gamma

    def forward(self,
                mu_ir, nu_ir, alpha_ir, beta_ir,
                mu_vis, nu_vis, alpha_vis, beta_vis):

        # ========= 1. 证据加权均值 =========
        nu_f = nu_ir + nu_vis + 1e-6

        mu_f = (nu_ir * mu_ir + nu_vis * mu_vis) / nu_f

        # ========= 2. α 融合 =========
        alpha_f = alpha_ir + alpha_vis+self.gamma
        # alpha_f = alpha_ir + alpha_vis

        # ========= 3. β 融合（带冲突项） =========
        beta_f = (
            beta_ir + beta_vis
            +self.gamma*nu_ir *((mu_ir - mu_f) ** 2)
            +self.gamma*nu_vis*((mu_vis - mu_f) ** 2)
        )
        # ========= 4. 融合不确定性 =========
        Ua_f = beta_f / (alpha_f - 1.0 + 1e-6)
        Ue_f = beta_f / (nu_f*(alpha_f - 1.0 + 1e-6))
        U_f = Ua_f+Ue_f

        # ========= 5. 最终融合 =========
        I_final = mu_f + U_f * (mu_ir - mu_vis)

        return I_final, mu_f, U_f


# ===============================
# 完整融合网络
# ===============================
class FusionModel(nn.Module):
    def __init__(self, gamma=0.5):
        super(FusionModel, self).__init__()

        self.ir_net = EvidenceRegressionNet()
        self.vis_net = EvidenceRegressionNet()
        self.fusion = NIGFusion(gamma=gamma)

    def forward(self, ir, vis):

        mu_ir, nu_ir, alpha_ir, beta_ir = self.ir_net(ir)
        mu_vis, nu_vis, alpha_vis, beta_vis = self.vis_net(vis)

        I_final, mu_f, U_f = self.fusion(
            mu_ir, nu_ir, alpha_ir, beta_ir,
            mu_vis, nu_vis, alpha_vis, beta_vis)

        return {
            "fused": I_final,
            "mu_f": mu_f,
            "U_f": U_f,
            "ir_params": (mu_ir, nu_ir, alpha_ir, beta_ir),
            "vis_params": (mu_vis, nu_vis, alpha_vis, beta_vis),
        }