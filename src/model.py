"""
Cognitive-Aim Core Model
Monocular depth estimation model based on cognitive science
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model, Dinov2Config
import math
import traceback

class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) Layer"""
    
    def __init__(self, in_features, out_features, rank=16, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
    

    def forward(self, x):
        # Simplified implementation, no longer depends on camera_indices and is_hard_sample variables
        # Directly apply linear transformation
        return self.lora_projection(x) * self.scaling

class AmbientStream(nn.Module):
    """Ambient awareness stream - processes overall scene information"""
    
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
    def forward(self, cls_token):
        """
        Args:
            cls_token: [batch_size, input_dim] - DINOv2 CLS token
        Returns:
            ambient_features: [batch_size, hidden_dim//4] - Global scene features
        """
        return self.mlp(cls_token)

class FocalStream(nn.Module):
    """Focal targeting stream - attention mechanism selects key regions (integrated curiosity-driven)"""
    
    def __init__(self, patch_dim, hidden_dim=256, num_heads=8, curiosity_guided=True, attention_dropout=0.1):
        super().__init__()
        self.patch_dim = patch_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.curiosity_guided = curiosity_guided
        
        # 自定义注意力机制 - 避免MultiheadAttention的均匀化问题
        self.query_proj = nn.Linear(patch_dim, patch_dim)
        self.key_proj = nn.Linear(patch_dim, patch_dim)
        self.value_proj = nn.Linear(patch_dim, patch_dim)
        self.scale = math.sqrt(patch_dim // num_heads)
        self.attention_dropout = nn.Dropout(0.0)
        
        # 好奇心驱动的注意力调制器
        if curiosity_guided:
            self.curiosity_modulator = nn.Sequential(
                nn.Linear(1, hidden_dim // 8),
                nn.ReLU(),
                nn.Linear(hidden_dim // 8, num_heads),
                nn.Sigmoid()
            )
        
        # 特征投影
        self.projection = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 4)
        )
        
        # 自适应权重学习
        self.adaptive_weight = nn.Parameter(torch.tensor(0.5))
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重以确保焦点流有足够的激活和注意力多样性"""
        # 初始化投影层
        for module in self.projection:
            if isinstance(module, nn.Linear):
                # 使用更保守的Xavier初始化
                nn.init.xavier_uniform_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)  # 零偏置
        
        # 初始化好奇心调制器
        if self.curiosity_guided:
            for module in self.curiosity_modulator:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.8)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
        
        # 初始化自定义注意力权重以增加多样性
        with torch.no_grad():
            # 初始化查询、键、值投影层
            nn.init.xavier_normal_(self.query_proj.weight, gain=2.0)
            nn.init.xavier_normal_(self.key_proj.weight, gain=2.0)
            nn.init.xavier_normal_(self.value_proj.weight, gain=1.0)
            
            # 偏置初始化
            if self.query_proj.bias is not None:
                nn.init.uniform_(self.query_proj.bias, -0.05, 0.05)
            if self.key_proj.bias is not None:
                nn.init.uniform_(self.key_proj.bias, -0.05, 0.05)
            if self.value_proj.bias is not None:
                nn.init.constant_(self.value_proj.bias, 0.0)
        
    def forward(self, patch_tokens, curiosity_score=None):
        """
        Args:
            patch_tokens: [batch_size, num_patches, patch_dim] - DINOv2的patch tokens
            curiosity_score: [batch_size] - 好奇心评分（可选）
        Returns:
            focal_features: [batch_size, hidden_dim//4] - 焦点特征
            attention_weights: [batch_size, num_patches] - 注意力权重
        """
        batch_size, num_patches, patch_dim = patch_tokens.size()
        
        # 添加2D位置编码
        def create_2d_position_encoding(num_patches, patch_dim, device):
            """创建2D位置编码，帮助模型理解空间关系"""
            # 确保位置编码的数量与实际patch数量完全匹配
            pos_encoding = torch.zeros(num_patches, patch_dim, device=device)
            
            # 尝试2D布局，如果不是完全平方数则使用1D
            patch_size = int(num_patches ** 0.5)
            if patch_size * patch_size == num_patches:
                # 2D位置编码（正方形布局）
                for i in range(num_patches):
                    row = i // patch_size
                    col = i % patch_size
                    
                    # 行位置编码（使用patch_dim的前半部分）
                    if patch_dim >= 4:  # 确保有足够的维度
                        div_term_row = torch.exp(torch.arange(0, patch_dim//2, 2, dtype=torch.float, device=device) * 
                                               -(math.log(10000.0) / (patch_dim//2)))
                        if len(div_term_row) > 0:
                            pos_encoding[i, 0:patch_dim//2:2] = torch.sin(row * div_term_row)
                            pos_encoding[i, 1:patch_dim//2:2] = torch.cos(row * div_term_row)
                        
                        # 列位置编码（使用patch_dim的后半部分）
                        div_term_col = torch.exp(torch.arange(0, patch_dim//2, 2, dtype=torch.float, device=device) * 
                                               -(math.log(10000.0) / (patch_dim//2)))
                        if len(div_term_col) > 0:
                            pos_encoding[i, patch_dim//2::2] = torch.sin(col * div_term_col)
                            pos_encoding[i, patch_dim//2+1::2] = torch.cos(col * div_term_col)
            else:
                # 1D位置编码（非正方形布局）
                position = torch.arange(0, num_patches, dtype=torch.float, device=device).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, patch_dim, 2, dtype=torch.float, device=device) * 
                                   -(math.log(10000.0) / patch_dim))
                if len(div_term) > 0:
                    pos_encoding[:, 0::2] = torch.sin(position * div_term)
                    if patch_dim > 1:
                        pos_encoding[:, 1::2] = torch.cos(position * div_term)
            
            return pos_encoding
        
        # 添加位置编码到patch tokens
        try:
            pos_encoding = create_2d_position_encoding(num_patches, patch_dim, patch_tokens.device)
            # 确保维度完全匹配
            if pos_encoding.shape[0] == num_patches and pos_encoding.shape[1] == patch_dim:
                patch_tokens = patch_tokens + pos_encoding.unsqueeze(0)  # 广播到batch维度
            else:
                print(f"位置编码维度不匹配，跳过: pos_encoding {pos_encoding.shape}, patch_tokens {patch_tokens.shape}")
        except Exception as e:
            print(f"位置编码创建失败，跳过: {e}")
        
        # 自定义注意力计算
        # 计算查询、键、值
        queries = self.query_proj(patch_tokens)  # [batch_size, num_patches, patch_dim]
        keys = self.key_proj(patch_tokens)       # [batch_size, num_patches, patch_dim]
        values = self.value_proj(patch_tokens)   # [batch_size, num_patches, patch_dim]
        
        # 计算注意力分数
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        
        # 应用softmax获得注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_patches, num_patches]
        attention_weights = self.attention_dropout(attention_weights)
        
        # 应用注意力权重到值
        attended_patches = torch.matmul(attention_weights, values)  # [batch_size, num_patches, patch_dim]
        
        # 计算每个patch的聚合注意力权重（用于可视化和后续处理）
        # 使用空间加权聚合策略，给中心区域更高的基础权重
        def create_center_bias_mask(num_patches, center_strength=0.3):
            """创建中心偏置掩码，增强中心区域的注意力权重"""
            # 根据实际patch数量计算patch_size
            patch_size = int(num_patches ** 0.5)
            if patch_size * patch_size != num_patches:
                # 如果不是完全平方数，使用1D中心偏置
                center_pos = num_patches // 2
                positions = torch.arange(num_patches, dtype=torch.float)
                distance = torch.abs(positions - center_pos)
                # 使用更集中的标准差
                sigma = num_patches / 12  # 从num_patches/8改为num_patches/12，使分布更集中
                center_bias = torch.exp(-distance**2 / (2 * sigma**2))
                return center_bias * center_strength
            
            # 2D中心偏置（正方形布局）
            center = patch_size // 2
            y, x = torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size), indexing='ij')
            # 计算距离中心的距离
            distance = torch.sqrt((x - center).float()**2 + (y - center).float()**2)
            # 创建更集中的高斯分布中心偏置（减小标准差）
            sigma = patch_size / 6  # 从patch_size/4改为patch_size/6，使分布更集中
            center_bias = torch.exp(-distance**2 / (2 * sigma**2))
            center_bias = center_bias.flatten()  # 展平为1D，长度正好是num_patches
            return center_bias * center_strength
        
        # 使用注意力权重矩阵的平均值作为基础
        patch_attention = attention_weights.mean(dim=1)  # [batch_size, num_patches]
        
        # 添加中心偏置
        num_patches = patch_attention.size(-1)
        center_bias = create_center_bias_mask(num_patches).to(patch_attention.device)
        patch_attention = patch_attention + center_bias.unsqueeze(0)  # 广播到batch维度
        
        # 如果基础权重仍然均匀，使用对角线元素
        if patch_attention.var() < 1e-6:
            patch_attention = torch.diagonal(attention_weights, dim1=-2, dim2=-1)  # [batch_size, num_patches]
            patch_attention = patch_attention + center_bias.unsqueeze(0)
        
        # 最后的备选：使用每行的最大值
        if patch_attention.var() < 1e-6:
            patch_attention, _ = torch.max(attention_weights, dim=-1)  # [batch_size, num_patches]
            patch_attention = patch_attention + center_bias.unsqueeze(0)
        
        # 最后的保险措施：基于输入特征的权重（不使用softmax归一化）
        if patch_attention.var() < 1e-6:
            # 基于输入patch tokens的L2范数作为重要性
            patch_norms = torch.norm(patch_tokens, dim=-1)  # [batch_size, num_patches]
            # 添加噪声增加变化
            noise = torch.randn_like(patch_norms) * 0.1 * patch_norms.std()
            patch_attention = patch_norms + noise
        
        # 使用更温和的归一化，保持更多的变化
        # 不使用softmax，而是简单的L1归一化
        patch_attention = patch_attention / (patch_attention.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 好奇驱动的注意力调制
        if self.curiosity_guided and curiosity_score is not None:
            # 将好奇心评分转换为注意力调制权重
            curiosity_modulation = self.curiosity_modulator(curiosity_score.unsqueeze(-1))  # [batch_size, num_heads]
            
            # 将好奇调制应用到patch注意力
            curiosity_weight = curiosity_modulation.mean(dim=-1, keepdim=True)  # [batch_size, 1]
            modulated_attention = patch_attention * (1.0 + curiosity_weight)  # [batch_size, num_patches]
            
            # 自适应权重混合原始注意力和调制后的注意力
            final_attention = (self.adaptive_weight * modulated_attention + 
                             (1 - self.adaptive_weight) * patch_attention)
        else:
            final_attention = patch_attention
        
        # 保持注意力权重的原始分布，不使用softmax避免过度平滑
        # final_attention = F.softmax(final_attention, dim=-1)  # 注释掉softmax
        # 确保权重为正值并归一化
        final_attention = torch.clamp(final_attention, min=1e-8)
        final_attention = final_attention / (final_attention.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 注意力熵正则化 - 鼓励注意力权重集中性
        # 计算注意力熵（训练时使用，推理时可忽略）
        if self.training:
            # 添加小的epsilon避免log(0)
            eps = 1e-8
            attention_entropy = -torch.sum(final_attention * torch.log(final_attention + eps), dim=-1)
            # 将熵损失存储为实例属性，供训练脚本访问
            # 修正：使用正熵作为损失（鼓励低熵，即更集中的注意力）
            entropy_loss = attention_entropy.mean()  # 正熵作为损失（鼓励集中注意力）
            self._last_attention_entropy = entropy_loss
            
            # 调试信息：检查注意力熵计算
            if hasattr(self, '_debug_entropy_count'):
                self._debug_entropy_count += 1
            else:
                self._debug_entropy_count = 1
            
            # 每100次前向传播打印一次调试信息
            if self._debug_entropy_count % 100 == 0:
                print(f"[调试] 注意力熵: {attention_entropy.mean().item():.6f}, 熵损失: {entropy_loss.item():.6f}, 权重方差: {final_attention.var().item():.2e}")
        else:
            self._last_attention_entropy = 0.0
        
        # 加权特征
        weighted_features = torch.sum(attended_patches * final_attention.unsqueeze(-1), dim=1)
        
        # 特征投影
        focal_features = self.projection(weighted_features)
        
        return focal_features, final_attention

class IterativeFocalStream(nn.Module):
    """迭代式焦点瞄准流 - 多步精细化瞄准（集成好奇心驱动）"""
    
    def __init__(self, patch_dim, hidden_dim=256, num_iterations=2, curiosity_guided=True, focus_strength=0.1, attention_dropout=0.1):
        super().__init__()
        self.num_iterations = num_iterations
        self.curiosity_guided = curiosity_guided
        self.focus_strength = focus_strength  # 聚焦强度参数
        
        # 多个焦点流（集成好奇心驱动）- 传递attention_dropout参数
        self.focal_streams = nn.ModuleList([
            FocalStream(patch_dim, hidden_dim, curiosity_guided=curiosity_guided, attention_dropout=attention_dropout) for _ in range(num_iterations)
        ])
        
        # 初始焦点（用于迭代焦点流）
        self.initial_focus = nn.Parameter(torch.randn(1, patch_dim))
        
        # 好奇心强度调节器（用于迭代间的动态调节）
        if curiosity_guided:
            self.curiosity_amplifier = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, num_iterations),
                nn.Softmax(dim=-1)
            )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim // 4 * num_iterations, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重以确保迭代焦点流有足够的激活和注意力多样性"""
        # 初始化融合层
        for module in self.fusion:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # 初始化好奇心放大器
        if self.curiosity_guided:
            for module in self.curiosity_amplifier:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.8)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
        
        # 初始焦点参数使用更小的标准差
        nn.init.normal_(self.initial_focus, mean=0.0, std=0.02)
        
        # 关键修复：为每个FocalStream实例重新初始化注意力权重
        for i, focal_stream in enumerate(self.focal_streams):
            with torch.no_grad():
                # 为每个迭代使用略微不同的初始化，增加多样性
                diversity_factor = 1.0 + 0.1 * i  # 每个迭代增加10%的多样性
                
                # 重新初始化自定义注意力权重
                if hasattr(focal_stream, 'query_proj'):
                    nn.init.xavier_normal_(focal_stream.query_proj.weight, gain=1.2 * diversity_factor)
                    nn.init.xavier_normal_(focal_stream.key_proj.weight, gain=1.2 * diversity_factor)
                    nn.init.xavier_normal_(focal_stream.value_proj.weight, gain=1.0 * diversity_factor)
                    
                    # 偏置初始化
                    if focal_stream.query_proj.bias is not None:
                        nn.init.uniform_(focal_stream.query_proj.bias, -0.01 * diversity_factor, 0.01 * diversity_factor)
                    if focal_stream.key_proj.bias is not None:
                        nn.init.uniform_(focal_stream.key_proj.bias, -0.01 * diversity_factor, 0.01 * diversity_factor)
                    if focal_stream.value_proj.bias is not None:
                        nn.init.constant_(focal_stream.value_proj.bias, 0.0)
        
    def forward(self, patch_tokens, curiosity_score=None):
        """
        Args:
            patch_tokens: [batch_size, num_patches, patch_dim]
            curiosity_score: [batch_size] - 好奇心评分（可选）
        Returns:
            fused_features: [batch_size, hidden_dim//4]
            attention_weights: [batch_size, num_patches] - 最后一次迭代的注意力权重
        """
        features_list = []
        attention_weights = None
        
        current_patches = patch_tokens
        
        # 好奇驱动的迭代权重分配
        if self.curiosity_guided and curiosity_score is not None:
            iteration_weights = self.curiosity_amplifier(curiosity_score.unsqueeze(-1))  # [batch_size, num_iterations]
        else:
            iteration_weights = None
        
        for i, focal_stream in enumerate(self.focal_streams):
            # 根据迭代位置调整好奇心强度
            if iteration_weights is not None:
                # 每次迭代使用不同的好奇心强度
                iter_curiosity = curiosity_score * iteration_weights[:, i]
            else:
                iter_curiosity = curiosity_score
            
            features, attn_weights = focal_stream(current_patches, iter_curiosity)
            features_list.append(features)
            attention_weights = attn_weights  # 保存最后一次迭代的注意力权重
            
            # 基于好奇注意力权重调整下一次迭代的patch tokens
            if i < len(self.focal_streams) - 1:  # 不是最后一次迭代
                # 使用注意力权重重新聚焦patch tokens
                enhanced_patches = current_patches * (1 + self.focus_strength * attn_weights.unsqueeze(-1))
                current_patches = enhanced_patches
        
        # 融合所有迭代的特征
        fused_features = self.fusion(torch.cat(features_list, dim=1))
        
        # 收集所有迭代的注意力熵损失
        if self.training:
            total_entropy_loss = 0.0
            entropy_count = 0
            for focal_stream in self.focal_streams:
                if hasattr(focal_stream, '_last_attention_entropy') and focal_stream._last_attention_entropy != 0:
                    total_entropy_loss += focal_stream._last_attention_entropy
                    entropy_count += 1
            
            # 计算平均熵损失并存储
            if entropy_count > 0:
                self._last_attention_entropy = total_entropy_loss / entropy_count
                # 调试信息
                if not hasattr(self, '_debug_iter_count'):
                    self._debug_iter_count = 0
                self._debug_iter_count += 1
                if self._debug_iter_count % 50 == 0:
                    print(f"[调试] IterativeFocalStream 平均熵损失: {self._last_attention_entropy:.6f}, 活跃流数: {entropy_count}/{len(self.focal_streams)}")
            else:
                self._last_attention_entropy = 0.0
        else:
            self._last_attention_entropy = 0.0
        
        return fused_features, attention_weights

class EXIFPriorDatabase(nn.Module):
    """EXIF经验库 - 编码相机参数和经验（匹配检查点结构）"""
    
    def __init__(self, num_cameras, hidden_dim=256):
        super().__init__()
        self.num_cameras = num_cameras
        
        # 相机模型嵌入 - 匹配检查点
        self.camera_embedding = nn.Embedding(num_cameras, 64)
        
        # 连续EXIF参数编码器 - 匹配检查点结构
        self.exif_encoder = nn.Sequential(
            nn.Linear(3, 64),  # focal_length, aperture, iso
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # 融合层 - 匹配检查点结构
        self.fusion = nn.Sequential(
            nn.Linear(128, hidden_dim),  # 64 + 64
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 4)
        )
        
    def forward(self, exif_data):
        """
        Args:
            exif_data: dict with keys 'camera_idx', 'focal_length', 'aperture', 'iso'
        Returns:
            exif_features: [batch_size, hidden_dim//4]
        """
        
        # 相机嵌入
        camera_idx = exif_data['camera_idx']
        if camera_idx.dim() > 1:
            camera_idx = camera_idx.squeeze(1)
        
        camera_features = self.camera_embedding(camera_idx)
        
        # 处理EXIF数据
        focal_length = exif_data['focal_length']
        aperture = exif_data['aperture']
        iso = exif_data['iso']
        
        # 确保维度正确
        if focal_length.dim() > 1:
            focal_length = focal_length.squeeze(1)
        if aperture.dim() > 1:
            aperture = aperture.squeeze(1)
        if iso.dim() > 1:
            iso = iso.squeeze(1)
        
        continuous_params = torch.stack([
            focal_length,
            aperture,
            torch.log(iso + 1)  # log变换ISO
        ], dim=1)
        exif_features = self.exif_encoder(continuous_params)
        
        # 融合
        combined_features = torch.cat([camera_features, exif_features], dim=1)
        return self.fusion(combined_features)

class CuriosityModule(nn.Module):
    """变分贝叶斯好奇心模块 - 基于VAE的不确定性估计"""
    
    def __init__(self, feature_dim, hidden_dim=128, enable_hierarchical=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.enable_hierarchical = enable_hierarchical
        self.latent_dim = feature_dim // 4  # 潜在空间维度
        
        # 变分编码器 (Encoder)
        self.encoder_mean = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, self.latent_dim)
        )
        
        self.encoder_logvar = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, self.latent_dim)
        )
        
        # 变分解码器 (Decoder)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, self.latent_dim)  # 重构到压缩维度
        )
        
        # 辅助不确定性估计网络
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Softplus()  # 确保输出为正值
        )
        
        # 分层好奇心模块
        if enable_hierarchical:
            # 几何一致性好奇心（与EXIF协同）
            self.geometric_curiosity = nn.Sequential(
                nn.Linear(feature_dim + 4, hidden_dim),  # 特征 + EXIF几何参数
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            
            # 局部探索好奇心（细节发现）
            self.local_curiosity = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            
            # 自适应权重学习（几何、局部、变分贝叶斯）
            self.curiosity_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.3]))
        
        # 探索历史记录（用于动态采样）
        self.register_buffer('exploration_history', torch.zeros(1000))  # 循环缓冲区
        self.register_buffer('history_pointer', torch.tensor(0))
        
    def forward(self, features, targets=None, exif_data=None, loss_type="robust", uncertainty_weight=0.1):
        """
        变分贝叶斯好奇心机制：通过VAE估计认知不确定性
        
        Args:
            features: [batch_size, feature_dim] - 模型特征
            targets: [batch_size] - 真实深度值（可选）
            exif_data: dict - EXIF信息（用于几何好奇心）
            loss_type: str - 损失类型 ("simple", "robust", "huber")
            uncertainty_weight: float - 不确定性权重
        Returns:
            curiosity_reward: [batch_size] - 综合好奇心奖励（非负值）
            uncertainty_score: [batch_size] - 不确定性评分（非负值）
            curiosity_components: dict - 各组件详情
        """
        batch_size = features.size(0)
        
        # 1. 变分编码：计算潜在空间的均值和方差
        mu = self.encoder_mean(features)  # [batch_size, latent_dim]
        logvar = self.encoder_logvar(features)  # [batch_size, latent_dim]
        
        # 2. 重参数化技巧：从潜在分布中采样
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # [batch_size, latent_dim]
        
        # 3. 变分解码：重构特征
        reconstructed = self.decoder(z)  # [batch_size, latent_dim]
        
        # 4. 计算重构误差（变分下界的重构项）
        target_features = features[:, :reconstructed.size(1)].detach()  # 取对应维度
        
        if loss_type == "simple":
            reconstruction_error = F.mse_loss(reconstructed, target_features, reduction='none').mean(dim=1)
        elif loss_type == "robust":
            diff = reconstructed - target_features
            reconstruction_error = torch.sqrt(torch.sum(diff ** 2, dim=1) + 1e-8)
            reconstruction_error = reconstruction_error / (1.0 + reconstruction_error)
        elif loss_type == "huber":
            diff = reconstructed - target_features
            abs_diff = torch.abs(diff)
            delta = 1.0
            huber_loss = torch.where(abs_diff <= delta, 0.5 * diff ** 2, delta * abs_diff - 0.5 * delta ** 2)
            reconstruction_error = huber_loss.mean(dim=1)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        # 5. 计算KL散度（变分下界的正则化项）
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # 6. 辅助不确定性估计
        uncertainty_estimate = self.uncertainty_head(features).squeeze(-1)
        
        # 7. 综合不确定性评分（确保非负）
        reconstruction_error = torch.clamp(reconstruction_error, min=0.0)
        kl_divergence = torch.clamp(kl_divergence, min=0.0)
        uncertainty_estimate = torch.clamp(uncertainty_estimate, min=0.0, max=10.0)
        
        # 变分贝叶斯不确定性：重构误差 + KL散度 + 辅助估计
        basic_uncertainty = reconstruction_error + 0.1 * kl_divergence + uncertainty_weight * uncertainty_estimate
        
        curiosity_components = {
            'reconstruction_error': reconstruction_error,
            'kl_divergence': kl_divergence,
            'uncertainty_estimate': uncertainty_estimate,
            'basic_uncertainty': basic_uncertainty,
            'latent_mean': mu,
            'latent_logvar': logvar
        }
        
        # 6. 分层好奇心计算（所有组件都确保非负）
        if self.enable_hierarchical and hasattr(self, 'geometric_curiosity'):
            # 几何一致性好奇心（基于EXIF先验）
            geometric_uncertainty = self._compute_geometric_curiosity(features, exif_data)
            
            # 局部探索好奇心（基于特征变化敏感性）
            local_uncertainty = self._compute_local_curiosity(features)
            
            # 权重归一化
            weights = F.softmax(self.curiosity_weights, dim=0)
            
            # 综合好奇心奖励（加权平均，确保非负）
            curiosity_reward = (weights[0] * geometric_uncertainty + 
                              weights[1] * local_uncertainty + 
                              weights[2] * basic_uncertainty)
            
            curiosity_components.update({
                'geometric_uncertainty': geometric_uncertainty,
                'local_uncertainty': local_uncertainty,
                'weights': weights
            })
            
            # 更新探索历史
            self._update_exploration_history(curiosity_reward.detach())
        else:
            curiosity_reward = basic_uncertainty
        
        # 7. 最终确保所有输出都是非负值
        curiosity_reward = torch.clamp(curiosity_reward, min=0.0, max=100.0)
        uncertainty_score = torch.clamp(basic_uncertainty, min=0.0, max=100.0)
        
        return curiosity_reward, uncertainty_score, curiosity_components
    
    def _compute_geometric_curiosity(self, features, exif_data):
        """计算几何一致性不确定性（变分贝叶斯版本）
        
        基于EXIF参数与视觉特征的一致性来评估几何不确定性。
        结合变分推断的思想，通过特征-EXIF联合分布的不确定性来量化几何理解程度。
        """
        batch_size = features.size(0)
        
        if exif_data is None:
            # 没有EXIF信息时，返回中等不确定性
            return torch.full((batch_size,), 0.5, device=features.device)
        
        try:
            # 组合EXIF几何参数（归一化处理）
            focal_length = exif_data.get('focal_length', torch.zeros(batch_size, device=features.device)).view(-1)
            aperture = exif_data.get('aperture', torch.zeros(batch_size, device=features.device)).view(-1)
            iso = exif_data.get('iso', torch.zeros(batch_size, device=features.device)).view(-1)
            
            # 归一化EXIF参数到[0,1]范围
            focal_length_norm = torch.clamp(focal_length / 200.0, 0.0, 1.0)  # 假设最大焦距200mm
            aperture_norm = torch.clamp(aperture / 32.0, 0.0, 1.0)  # 假设最大光圈f/32
            iso_norm = torch.clamp(iso / 6400.0, 0.0, 1.0)  # 假设最大ISO 6400
            
            exif_features = torch.stack([
                focal_length_norm,
                aperture_norm, 
                iso_norm,
                torch.ones(batch_size, device=features.device)  # 偏置项
            ], dim=1)
            
            combined_features = torch.cat([features, exif_features], dim=1)
            geometric_uncertainty = self.geometric_curiosity(combined_features).squeeze(-1)
            
            # Sigmoid输出已经在[0,1]范围内，表示不确定性程度
            return torch.clamp(geometric_uncertainty, min=0.0, max=1.0)
            
        except Exception as e:
            print(f"几何不确定性计算失败: {e}")
            # 计算失败时返回高不确定性
            return torch.full((batch_size,), 0.8, device=features.device)
    
    def _compute_local_curiosity(self, features):
        """计算局部探索不确定性（变分贝叶斯版本）
        
        通过特征扰动测试模型对局部变化的敏感性。
        结合变分推断思想，高敏感性表明潜在空间的不确定性，反映模型理解的不稳定性。
        """
        # 基础局部不确定性评估
        local_uncertainty_base = self.local_curiosity(features).squeeze(-1)
        
        # 特征扰动敏感性测试
        with torch.no_grad():
            # 添加小幅随机扰动
            noise_scale = 0.01
            noise = torch.randn_like(features) * noise_scale
            noisy_features = features + noise
            
            # 计算扰动后的不确定性
            noisy_uncertainty = self.local_curiosity(noisy_features).squeeze(-1)
            
            # 敏感性 = 扰动前后的差异程度
            sensitivity = torch.abs(local_uncertainty_base - noisy_uncertainty)
            
        # 综合局部不确定性：基础不确定性 + 敏感性权重
        # 敏感性高的区域表明模型理解不稳定，应该增加不确定性
        local_uncertainty = local_uncertainty_base + sensitivity * 0.2
        
        # 确保输出在合理范围内（Sigmoid + 敏感性调整）
        return torch.clamp(local_uncertainty, min=0.0, max=1.0)
    
    def _update_exploration_history(self, rewards):
        """更新探索历史记录"""
        if not hasattr(self, 'exploration_history') or self.exploration_history is None:
            return
        
        # 展平rewards并更新历史
        if rewards.dim() > 1:
            rewards = rewards.mean(dim=-1)
        
        with torch.no_grad():
            for reward in rewards:
                idx = int(self.history_pointer) % self.exploration_history.size(0)
                self.exploration_history[idx] = reward.item()
                self.history_pointer = (self.history_pointer + 1) % self.exploration_history.size(0)
    
    def get_exploration_statistics(self):
        """获取探索统计信息"""
        if not hasattr(self, 'exploration_history') or self.exploration_history is None:
            return {'mean': 0., 'std': 0., 'max': 0., 'samples': 0}
        
        with torch.no_grad():
            # 过滤掉未初始化的值（假设初始为0）
            valid_samples = self.exploration_history[self.exploration_history > 0]
            
            if len(valid_samples) == 0:
                return {'mean': 0., 'std': 0., 'max': 0., 'samples': 0}
            
            return {
                'mean': float(valid_samples.mean()),
                'std': float(valid_samples.std()) if len(valid_samples) > 1 else 0.,
                'max': float(valid_samples.max()),
                'min': float(valid_samples.min()),
                'samples': int(len(valid_samples))
            }

class CognitiveAimModel(nn.Module):
    """Cognitive-Aim 主模型"""
    
    def __init__(self, config, camera_info=None):
        super().__init__()
        self.config = config
        
        # DINOv2 骨干网络
        self.backbone_size = config.get('backbone_size', 'base')
        if self.backbone_size == 'base':
            model_name = 'facebook/dinov2-base'
            self.feature_dim = 768
        elif self.backbone_size == 'large':
            model_name = 'facebook/dinov2-large'
            self.feature_dim = 1024
        else:
            model_name = 'facebook/dinov2-base'
            self.feature_dim = 768
            
        self.backbone = Dinov2Model.from_pretrained(model_name)
        
        # 冻结骨干网络（可选）
        if config.get('freeze_backbone', True):
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # LoRA适配（可选）
        self.use_lora = config.get('use_lora', False)
        if self.use_lora:
            self.lora_layers = nn.ModuleList()
            for layer in self.backbone.encoder.layer:
                lora = LoRALayer(
                    self.feature_dim, 
                    self.feature_dim, 
                    rank=config.get('lora_rank', 16)
                )
                self.lora_layers.append(lora)
        
        # 认知模块（支持多层级配置读取）
        # 优先从model.cognitive_modules读取，其次从顶层cognitive_modules读取
        model_config = config.get('model', {})
        cognitive_modules = model_config.get('cognitive_modules', config.get('cognitive_modules', []))
        debug_init = config.get('debug_init', False)
        
        if debug_init:
            print(f"配置中的认知模块: {cognitive_modules}")
            print(f"从路径读取: {'model.cognitive_modules' if 'cognitive_modules' in model_config else 'cognitive_modules'}")
        
        # 全局感知流
        if 'ambient_stream' in cognitive_modules:
            self.ambient_stream = AmbientStream(self.feature_dim)
            self.use_ambient = True
            if debug_init:
                print("全局感知流已启用")
        else:
            self.use_ambient = False
        
        # 焦点瞄准流（支持好奇心驱动）
        if 'iterative_focal_stream' in cognitive_modules:
            curiosity_guided = config.get('curiosity_guided_attention', {}).get('enabled', False)
            focal_config = config.get('focal_config', {})
            attention_dropout = config.get('curiosity_guided_attention', {}).get('attention_dropout', 0.1)
            self.focal_stream = IterativeFocalStream(
                self.feature_dim,
                hidden_dim=config.get('focal_hidden_dim', 256),
                num_iterations=focal_config.get('num_iterations', 3),
                curiosity_guided=curiosity_guided,
                focus_strength=focal_config.get('focus_strength', 1.5),
                attention_dropout=attention_dropout
            )
            self.use_focal = True
            self.use_iterative = True
            if debug_init:
                print(f"迭代式焦点瞄准流已启用（{focal_config.get('num_iterations', 3)}次迭代，dropout={attention_dropout}）")
        elif 'focal_stream' in cognitive_modules:
            attention_dropout = config.get('curiosity_guided_attention', {}).get('attention_dropout', 0.1)
            self.focal_stream = FocalStream(self.feature_dim, attention_dropout=attention_dropout)
            self.use_focal = True
            self.use_iterative = False
            if debug_init:
                print(f"基础焦点瞄准流已启用（dropout={attention_dropout}）")
        else:
            self.use_focal = False
            self.use_iterative = False
        
        # EXIF经验库
        if 'exif_prior_database' in cognitive_modules and camera_info:
            self.exif_prior = EXIFPriorDatabase(camera_info['num_cameras'])
            self.use_exif = True
            if debug_init:
                print("EXIF经验库已启用")
        else:
            self.use_exif = False
        
        # 计算融合特征维度 - 匹配检查点结构
        # 注意：各个认知模块的实际输出维度是 hidden_dim // 4 = 256 // 4 = 64
        module_output_dim = 256 // 4  # 64维
        fusion_dim = 0
        if self.use_ambient:
            fusion_dim += module_output_dim  # AmbientStream 输出维度为 64
        if self.use_focal:
            fusion_dim += module_output_dim  # FocalStream 输出维度为 64
        if self.use_exif:
            fusion_dim += module_output_dim  # EXIFPriorDatabase 输出维度为 64
        
        if fusion_dim == 0:
            fusion_dim = self.feature_dim  # 如果没有认知模块，直接使用CLS token
            
        # 修正：使用检查点中的融合维度192来匹配已训练的权重
        checkpoint_fusion_dim = 192  # 检查点中的融合层维度
        self.fusion_dim = checkpoint_fusion_dim
        
        # 融合层 - 匹配检查点维度
        self.fusion = nn.Sequential(
            nn.Linear(checkpoint_fusion_dim, checkpoint_fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 打印融合层的输入维度，用于调试
        print(f"融合层输入维度: {self.fusion_dim}")
        
        # 维度对齐器初始化 - 根据实际特征输出维度配置
        self.target_fusion_dim = 768
        # 所有认知模块输出都是64维 (hidden_dim//4)
        self.ambient_dim_aligner = DimensionAligner(target_dim=self.target_fusion_dim, source_dim=module_output_dim)
        self.focal_dim_aligner   = DimensionAligner(target_dim=self.target_fusion_dim, source_dim=module_output_dim)
        self.exif_dim_aligner    = DimensionAligner(target_dim=self.target_fusion_dim, source_dim=module_output_dim)

        # 决策头（匹配融合层输出维度192）- 使用Softplus确保正深度值
        self.decision_head = nn.Sequential(
            nn.Linear(checkpoint_fusion_dim, 1),
            nn.Softplus()  # 平滑的正值激活函数，避免梯度消失
        )
        
        # 初始化决策头权重以确保稳定的梯度和合理的深度范围
        with torch.no_grad():
            nn.init.xavier_uniform_(self.decision_head[0].weight, gain=1.0)
            nn.init.constant_(self.decision_head[0].bias, 1.0)  # 更大的正偏置确保合理深度范围
        
        # 置信度头（匹配融合层输出维度192）
        self.confidence_head = nn.Sequential(
            nn.Linear(checkpoint_fusion_dim, 1),
            nn.ReLU(),  # 先使用ReLU激活
            nn.Linear(1, 1),
            nn.Sigmoid()  # 确保置信度在0-1范围内
        )
        
        # 初始化置信度头的偏置，使其倾向于产生较高的置信度
        with torch.no_grad():
            self.confidence_head[2].bias.fill_(2.0)  # 添加正偏置
        
        # 增强好奇模块（恢复原始配置以匹配检查点）
        self.curiosity_module = CuriosityModule(
            self.target_fusion_dim, 
            hidden_dim=256,  # 恢复原始隐藏层维度
            enable_hierarchical=config.get('enable_hierarchical_curiosity', True)
        )
        print("[调试] 好奇心模块已启用（隐藏维度256，匹配检查点）")
        
        # 全局融合对齐器 - 动态适配连接后的特征维度
        # 如果3个模块都启用：3*768=2304维，如果2个模块：2*768=1536维
        max_source_dim = self.target_fusion_dim * 3  # 最大可能的源维度
        self.global_aligner = DimensionAligner(target_dim=self.target_fusion_dim, source_dim=max_source_dim)
        
    def get_features_aligned(self, images, exif_data=None):
        """提取特征并实现自动维度对齐用于认知架构内部使用
        
        解决192×192与32×128矩阵乘法维度不匹配问题
        
        Args:
            images: [batch_size, 3, H, W]
            exif_data: dict with EXIF information
        Returns:
            fused_features: [batch_size, fusion_dim] - 统一维度后的特征
        """
        try:
            # 提取DINOv2特征
            outputs = self.backbone(images, output_hidden_states=True)
            cls_token = outputs.last_hidden_state[:, 0]  # [B, hidden_dim]
            patch_tokens = outputs.last_hidden_state[:, 1:]  # [B, N, hidden_dim]
        
            # 认知模块处理
            cognitive_features = []
            feature_dims = []
            
            # 1. 全局感知流（CLS token → 64维对齐到768维）
            if self.use_ambient:
                ambient_raw = self.ambient_stream(cls_token)
                ambient_aligned = self.ambient_dim_aligner(ambient_raw)
                cognitive_features.append(ambient_aligned)
                feature_dims.append(768)
        
            # 2. 焦点瞄准流（patch tokens → 64维对齐到768维）
            if self.use_focal:
                # 计算初步好奇心评分（基于CLS token）
                curiosity_score = None
                if hasattr(self, 'curiosity_module'):
                    try:
                        cls_features = cls_token.view(cls_token.size(0), -1)
                        initial_curiosity, _, _ = self.curiosity_module(cls_features)
                        curiosity_score = initial_curiosity.mean(dim=-1) if initial_curiosity.dim() > 1 else initial_curiosity
                    except Exception as e:
                        print(f"初步好奇心计算失败，使用默认值: {e}")
                        # 提供默认好奇心评分
                        batch_size = cls_token.size(0)
                        curiosity_score = torch.ones(batch_size, device=cls_token.device) * 0.5
                else:
                    # 如果没有好奇心模块，提供默认评分
                    batch_size = cls_token.size(0)
                    curiosity_score = torch.ones(batch_size, device=cls_token.device) * 0.5
                
                # 统一使用focal_stream，无论是IterativeFocalStream还是FocalStream都赋值给了这个属性
                focal_raw, attn_weights = self.focal_stream(patch_tokens, curiosity_score)
                
                focal_aligned = self.focal_dim_aligner(focal_raw)
                cognitive_features.append(focal_aligned)
                feature_dims.append(768)
        
            # 直接获取原始64维特征以匹配检查点结构
            # 跳过768维对齐逻辑，直接使用64维特征
            raw_features = []
            if self.use_ambient:
                ambient_raw = self.ambient_stream(cls_token)  # 64维
                raw_features.append(ambient_raw)
            if self.use_focal:
                focal_raw, _ = self.focal_stream(patch_tokens, curiosity_score)  # 64维
                raw_features.append(focal_raw)
            if self.use_exif and exif_data is not None:
                exif_raw = self.exif_prior(exif_data)  # 64维
                raw_features.append(exif_raw)
            
            # 处理特征维度不匹配的情况
            if len(raw_features) == 0:
                raise RuntimeError("认知模块未返回任何特征！")
            
            # 连接64维特征，如果缺少EXIF模块则用零填充以匹配检查点的192维
            concatenated_features = torch.cat(raw_features, dim=1)  # [B, 64*n]
            
            # 如果特征维度不足192（缺少EXIF模块），用零填充
            if concatenated_features.size(1) < 192:
                batch_size = concatenated_features.size(0)
                padding_size = 192 - concatenated_features.size(1)
                padding = torch.zeros(batch_size, padding_size, device=concatenated_features.device)
                concatenated_features = torch.cat([concatenated_features, padding], dim=1)
                print(f"[调试] 缺少EXIF模块，用零填充到192维")
            
            print(f"[调试] 连接后特征维度: {concatenated_features.shape}")
            
            # 通过融合层处理以匹配检查点结构
            fused_features = self.fusion(concatenated_features)
            print(f"[调试] 融合后特征维度: {fused_features.shape}")
            
            return fused_features
        
        except Exception as e:
            print(f"get_features_aligned异常: {e}")
            traceback.print_exc()
            # 返回备用特征
            backup_features = torch.zeros(images.size(0), self.fusion_dim, device=images.device)
            print(f"返回备用特征维度: {backup_features.shape}")
            return backup_features
    
    def get_attention_weights(self):
        """获取最后一次前向传播的注意力权重"""
        if hasattr(self, '_last_attention_weights'):
            return self._last_attention_weights
        return None

    def forward(self, images, exif_data=None, return_attention=False):
        """前向传播函数 - 集成维度对齐的认知架构全流程
        
        Args:
            images: [batch_size, 3, H, W] - 输入图像
            exif_data: dict with EXIF 信息  
            return_attention: bool - 是否返回注意力权重
            
        Returns:
            depth_pred: [batch_size, 1] - 深度预测
            confidence: [batch_size, 1] - 置信度
            attention_weights: (可选) 注意力权重
        """
        # 使用新的对齐特征提取方法
        aligned_features = self.get_features_aligned(images, exif_data)
        
        if aligned_features is None:
            raise ValueError(f"特征对齐结果异常: None")
        
        # 特征维度检查
        if aligned_features.shape[1] != 192:
            raise ValueError(f"特征维度不匹配，期望192维，实际{aligned_features.shape[1]}维")

        # 统一应用对齐后的特征进行预测
        fused_features = aligned_features  # 已经过对齐处理
        self.fusion_features = fused_features  # 存储融合特征以供好奇心模块使用
        
        # 获取并保存注意力权重（用于可视化）
        # 只有在没有已保存的引导注意力权重时才设置基础权重
        if self.use_focal and not hasattr(self, '_last_attention_weights'):
            try:
                outputs = self.backbone(images, output_hidden_states=True)
                cls_token = outputs.last_hidden_state[:, 0]
                patch_tokens = outputs.last_hidden_state[:, 1:]
                
                batch_size = patch_tokens.size(0)
                device = patch_tokens.device
                if hasattr(self, 'curiosity_module'):
                    try:
                        cls_features = cls_token.view(cls_token.size(0), -1)
                        initial_curiosity, _, _ = self.curiosity_module(cls_features)
                        curiosity_score = initial_curiosity.mean(dim=-1) if initial_curiosity.dim() > 1 else initial_curiosity
                        curiosity_score = torch.clamp(curiosity_score, 0.5, 1.0)
                    except Exception as e:
                        curiosity_score = torch.ones(batch_size, device=device) * 1.0
                else:
                    curiosity_score = torch.ones(batch_size, device=device) * 1.0
                
                _, attn_weights = self.focal_stream(patch_tokens, curiosity_score)
                self._last_attention_weights = attn_weights
            except Exception as e:
                self._last_attention_weights = None
        elif not self.use_focal:
            self._last_attention_weights = None
        
        # 深度预测（使用标准化的特征维度）
        depth_pred = self.decision_head(fused_features)
        
        # 置信度预测（同样使用对齐后的特征）
        confidence = self.confidence_head(fused_features)
        
        # 返回结果格式统一
        if return_attention:
            if self.use_focal:
                # 获取注意力权重需要重新用原始forward
                outputs = self.backbone(images, output_hidden_states=True)
                cls_token = outputs.last_hidden_state[:, 0]
                patch_tokens = outputs.last_hidden_state[:, 1:]
                
                # 使用实际的好奇心评分而非硬编码值
                batch_size = patch_tokens.size(0)
                device = patch_tokens.device
                if hasattr(self, 'curiosity_module'):
                    try:
                        cls_features = cls_token.view(cls_token.size(0), -1)
                        initial_curiosity, _, _ = self.curiosity_module(cls_features)
                        curiosity_score = initial_curiosity.mean(dim=-1) if initial_curiosity.dim() > 1 else initial_curiosity
                        # 确保好奇心评分在合理范围内
                        curiosity_score = torch.clamp(curiosity_score, 0.5, 1.0)
                    except Exception as e:
                        print(f"好奇心模块计算失败，使用最大值: {e}")
                        curiosity_score = torch.ones(batch_size, device=device) * 1.0
                else:
                    curiosity_score = torch.ones(batch_size, device=device) * 1.0
                
                _, attn_weights = self.focal_stream(patch_tokens, curiosity_score)
                return depth_pred, confidence, attn_weights
            else:
                # 如果没有焦点流，返回None作为注意力权重
                return depth_pred, confidence, None
        
        return depth_pred, confidence
    
    def forward_with_guidance(self, images, exif_data=None, attention_guidance=None, return_attention=False):
        """支持用户指令引导的前向传播函数
        
        Args:
            images: [batch_size, 3, H, W] - 输入图像
            exif_data: dict with EXIF 信息
            attention_guidance: [196] - 用户指令生成的注意力引导权重
            return_attention: bool - 是否返回注意力权重
            
        Returns:
            depth_pred: [batch_size, 1] - 深度预测
            confidence: [batch_size, 1] - 置信度
            attention_weights: (可选) 注意力权重
        """
        try:
            # 提取DINOv2特征
            outputs = self.backbone(images, output_hidden_states=True)
            cls_token = outputs.last_hidden_state[:, 0]  # [B, hidden_dim]
            patch_tokens = outputs.last_hidden_state[:, 1:]  # [B, N, hidden_dim]
        
            # 认知模块处理 - 使用与forward方法相同的192维融合逻辑
            attention_weights = None
            
            # 计算初步好奇心评分（基于CLS token）
            curiosity_score = None
            if hasattr(self, 'curiosity_module'):
                try:
                    cls_features = cls_token.view(cls_token.size(0), -1)
                    initial_curiosity, _, _ = self.curiosity_module(cls_features)
                    curiosity_score = initial_curiosity.mean(dim=-1) if initial_curiosity.dim() > 1 else initial_curiosity
                except Exception as e:
                    print(f"初步好奇心计算失败，使用默认值: {e}")
                    curiosity_score = None
            
            # 获取原始64维特征以匹配检查点结构
            raw_features = []
            
            # 1. 全局感知流（CLS token → 64维）
            if self.use_ambient:
                ambient_raw = self.ambient_stream(cls_token)  # 64维
                raw_features.append(ambient_raw)
        
            # 2. 焦点瞄准流（patch tokens → 64维）- 应用用户引导
            if self.use_focal:
                # 应用用户指令引导的焦点瞄准
                if attention_guidance is not None:
                    focal_raw, attn_weights = self._guided_focal_stream(
                        patch_tokens, curiosity_score, attention_guidance
                    )
                else:
                    focal_raw, attn_weights = self.focal_stream(patch_tokens, curiosity_score)
                
                raw_features.append(focal_raw)  # 64维
                attention_weights = attn_weights
                # 保存注意力权重供可视化使用
                self._last_attention_weights = attn_weights
        
            # 3. EXIF经验库（64维）  
            if self.use_exif and exif_data is not None:
                exif_raw = self.exif_prior(exif_data)  # 64维
                raw_features.append(exif_raw)
        
            # 连接64维特征：3个模块 = 192维，匹配检查点
            if len(raw_features) == 0:
                raise RuntimeError("认知模块未返回任何特征！")
            
            concatenated_features = torch.cat(raw_features, dim=1)  # [B, 64*n]
            
            # 通过融合层处理以匹配检查点结构
            fused_features = self.fusion(concatenated_features)
            
            # 直接使用融合后的特征进行预测
            depth_pred = self.decision_head(fused_features)
            confidence = self.confidence_head(fused_features)
            
            if return_attention:
                return depth_pred, confidence, attention_weights
            else:
                return depth_pred, confidence
            
        except Exception as e:
            print(f"引导式特征提取失败: {e}")
            # 回退到标准前向传播
            return self.forward(images, exif_data, return_attention)
    
    def _guided_focal_stream(self, patch_tokens, curiosity_score, attention_guidance):
        """应用用户指令引导的焦点瞄准流
        
        Args:
            patch_tokens: [batch_size, num_patches, patch_dim]
            curiosity_score: [batch_size] - 好奇心评分
            attention_guidance: [num_patches] - 用户指令引导权重
            
        Returns:
            focal_features: [batch_size, hidden_dim//4]
            attention_weights: [batch_size, num_patches]
        """
        batch_size = patch_tokens.size(0)
        
        # 获取基础注意力权重
        base_features, base_attention = self.focal_stream(patch_tokens, curiosity_score)
        
        # 应用用户指令引导
        if attention_guidance is not None:
            # 处理字符串类型的指导指令
            if isinstance(attention_guidance, str):
                # 将字符串指令转换为空间注意力权重
                num_patches = patch_tokens.size(1)
                patch_size = int(math.sqrt(num_patches))  # 假设是方形patch布局
                
                # 创建空间注意力掩码
                spatial_mask = torch.ones(patch_size, patch_size, device=patch_tokens.device)
                
                if attention_guidance.lower() == 'center':
                    # 中心区域增强
                    center_y, center_x = patch_size // 2, patch_size // 2
                    radius = max(1, patch_size // 4)
                    for y in range(patch_size):
                        for x in range(patch_size):
                            dist = math.sqrt((y - center_y)**2 + (x - center_x)**2)
                            if dist <= radius:
                                spatial_mask[y, x] = 3.0  # 中心区域权重增强
                            elif dist <= radius * 2:
                                spatial_mask[y, x] = 1.5  # 中等权重
                                
                elif attention_guidance.lower() == 'left':
                    # 左侧聚焦点（使用圆形聚焦模式）
                    focus_y, focus_x = patch_size // 2, patch_size // 4  # 左侧1/4位置
                    radius = max(1, patch_size // 6)  # 更小的聚焦半径
                    for y in range(patch_size):
                        for x in range(patch_size):
                            dist = math.sqrt((y - focus_y)**2 + (x - focus_x)**2)
                            if dist <= radius:
                                spatial_mask[y, x] = 5.0  # 强聚焦
                            elif dist <= radius * 2:
                                spatial_mask[y, x] = 2.0  # 中等权重
                    
                elif attention_guidance.lower() == 'right':
                    # 右侧聚焦点（使用圆形聚焦模式）
                    focus_y, focus_x = patch_size // 2, patch_size * 3 // 4  # 右侧3/4位置
                    radius = max(1, patch_size // 6)  # 更小的聚焦半径
                    for y in range(patch_size):
                        for x in range(patch_size):
                            dist = math.sqrt((y - focus_y)**2 + (x - focus_x)**2)
                            if dist <= radius:
                                spatial_mask[y, x] = 5.0  # 强聚焦
                            elif dist <= radius * 2:
                                spatial_mask[y, x] = 2.0  # 中等权重
                    
                elif attention_guidance.lower() == 'top':
                    # 上方聚焦点（使用圆形聚焦模式）
                    focus_y, focus_x = patch_size // 4, patch_size // 2  # 上方1/4位置
                    radius = max(1, patch_size // 6)  # 更小的聚焦半径
                    for y in range(patch_size):
                        for x in range(patch_size):
                            dist = math.sqrt((y - focus_y)**2 + (x - focus_x)**2)
                            if dist <= radius:
                                spatial_mask[y, x] = 5.0  # 强聚焦
                            elif dist <= radius * 2:
                                spatial_mask[y, x] = 2.0  # 中等权重
                    
                elif attention_guidance.lower() == 'bottom':
                    # 下方聚焦点（使用圆形聚焦模式）
                    focus_y, focus_x = patch_size * 3 // 4, patch_size // 2  # 下方3/4位置
                    radius = max(1, patch_size // 6)  # 更小的聚焦半径
                    for y in range(patch_size):
                        for x in range(patch_size):
                            dist = math.sqrt((y - focus_y)**2 + (x - focus_x)**2)
                            if dist <= radius:
                                spatial_mask[y, x] = 5.0  # 强聚焦
                            elif dist <= radius * 2:
                                spatial_mask[y, x] = 2.0  # 中等权重
                                
                elif attention_guidance.lower() in ['top-left', 'topleft']:
                    # 左上角聚焦点
                    focus_y, focus_x = patch_size // 4, patch_size // 4  # 左上角1/4位置
                    radius = max(1, patch_size // 6)  # 更小的聚焦半径
                    for y in range(patch_size):
                        for x in range(patch_size):
                            dist = math.sqrt((y - focus_y)**2 + (x - focus_x)**2)
                            if dist <= radius:
                                spatial_mask[y, x] = 5.0  # 强聚焦
                            elif dist <= radius * 2:
                                spatial_mask[y, x] = 2.0  # 中等权重
                                
                elif attention_guidance.lower() in ['top-right', 'topright']:
                    # 右上角聚焦点
                    focus_y, focus_x = patch_size // 4, patch_size * 3 // 4  # 右上角位置
                    radius = max(1, patch_size // 6)  # 更小的聚焦半径
                    for y in range(patch_size):
                        for x in range(patch_size):
                            dist = math.sqrt((y - focus_y)**2 + (x - focus_x)**2)
                            if dist <= radius:
                                spatial_mask[y, x] = 5.0  # 强聚焦
                            elif dist <= radius * 2:
                                spatial_mask[y, x] = 2.0  # 中等权重
                                
                elif attention_guidance.lower() in ['bottom-left', 'bottomleft']:
                    # 左下角聚焦点
                    focus_y, focus_x = patch_size * 3 // 4, patch_size // 4  # 左下角位置
                    radius = max(1, patch_size // 6)  # 更小的聚焦半径
                    for y in range(patch_size):
                        for x in range(patch_size):
                            dist = math.sqrt((y - focus_y)**2 + (x - focus_x)**2)
                            if dist <= radius:
                                spatial_mask[y, x] = 5.0  # 强聚焦
                            elif dist <= radius * 2:
                                spatial_mask[y, x] = 2.0  # 中等权重
                                
                elif attention_guidance.lower() in ['bottom-right', 'bottomright']:
                    # 右下角聚焦点
                    focus_y, focus_x = patch_size * 3 // 4, patch_size * 3 // 4  # 右下角位置
                    radius = max(1, patch_size // 6)  # 更小的聚焦半径
                    for y in range(patch_size):
                        for x in range(patch_size):
                            dist = math.sqrt((y - focus_y)**2 + (x - focus_x)**2)
                            if dist <= radius:
                                spatial_mask[y, x] = 5.0  # 强聚焦
                            elif dist <= radius * 2:
                                spatial_mask[y, x] = 2.0  # 中等权重
                
                # 将2D掩码展平为1D
                attention_guidance = spatial_mask.flatten()
                
                # 继续处理转换后的数值型引导权重
                pass
            
            # 确保引导权重与patch数量匹配
            num_patches = patch_tokens.size(1)
            if attention_guidance.size(0) != num_patches:
                # 如果维度不匹配，进行插值调整
                guidance_size = int(math.sqrt(attention_guidance.size(0)))
                target_size = int(math.sqrt(num_patches))
                
                guidance_2d = attention_guidance.view(guidance_size, guidance_size)
                guidance_2d = F.interpolate(
                    guidance_2d.unsqueeze(0).unsqueeze(0),
                    size=(target_size, target_size),
                    mode='bilinear',
                    align_corners=False
                )
                attention_guidance = guidance_2d.squeeze().flatten()
            
            # 扩展到批次维度
            guidance_batch = attention_guidance.unsqueeze(0).expand(batch_size, -1)
            
            # 使用加权融合而不是简单相乘，以保持更强的引导效果
            alpha = 0.7  # 引导权重的影响强度
            guided_attention = alpha * guidance_batch + (1 - alpha) * base_attention
            
            # 使用温度缩放的softmax来保持更多对比度
            temperature = 0.05  # 更低的温度以增强对比度
            guided_attention = F.softmax(guided_attention / temperature, dim=-1)
            
            # 使用引导后的注意力重新计算特征
            weighted_patches = torch.sum(
                patch_tokens * guided_attention.unsqueeze(-1), dim=1
            )
            
            # 特征投影 - 确保输出64维以匹配其他认知模块
            if hasattr(self.focal_stream, 'projection'):
                guided_features = self.focal_stream.projection(weighted_patches)
            else:
                # 如果没有投影层，创建一个临时的线性层，输出64维
                temp_projection = nn.Linear(self.feature_dim, 64).to(patch_tokens.device)
                guided_features = temp_projection(weighted_patches)
            
            return guided_features, guided_attention
        else:
            return base_features, base_attention
        
    def get_features(self, images, exif_data=None):
        """保持向后兼容的特征提取接口"""
        return self.get_features_aligned(images, exif_data)

    def compute_curiosity_loss(self, features, depth_targets=None, exif_data=None, loss_type="robust", uncertainty_weight=0.1):
        """计算增强好奇tops损失
        
        Args:
            features: [batch_size, target_fusion_dim] - 融合特征
            depth_targets: [batch_size] - 目标深度值（可选）
            exif_data: dict - EXIF信息（用于几何好奇tops）
            loss_type: str - 损失类型
            uncertainty_weight: float - 不确定性权重
            
        Returns:
            tuple: (curiosity_reward, curiosity_components)
        """
        try:
            curiosity_reward, prediction_error, curiosity_components = self.curiosity_module(
                features, depth_targets, exif_data, loss_type, uncertainty_weight
            )
            return curiosity_reward, curiosity_components
            
        except Exception as e:
            print(f"增强好奇tops损失计算失败: {e}")
            batch_size = features.size(0)
            return torch.zeros(batch_size, device=features.device), {}

    def get_exploration_stats(self):
        """获取探索统计信息"""
        if hasattr(self, 'curiosity_module'):
            return self.curiosity_module.get_exploration_statistics()
        return {'mean': 0., 'std': 0., 'max': 0., 'samples': 0}


# 文件末尾 - 重复的create_model函数已删除
class DimensionAligner(nn.Module):
    """自动对齐不同维度的特征向量"""
    
    def __init__(self, target_dim, source_dim=None):
        super().__init__()
        self.target_dim = target_dim
        self.source_dim = source_dim
        
        # 如果没有指定源维度，使用自适应投影
        if source_dim is None:
            self.projection = None  # 延迟初始化
        else:
            self.projection = nn.Linear(source_dim, target_dim) if target_dim != source_dim else nn.Identity()
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 [batch_size, ..., source_dim] 或 [batch_size, ..., target_dim]
        Returns:
            对齐后的特征 [batch_size, ..., target_dim]
        """
        original_shape = x.shape
        batch_size = x.shape[0]
        
        # 展平除批次维度外的所有维度为二维 [batch_size, features]
        if len(x.shape) > 2:
            # 计算除批次维度外的总特征数
            num_features = 1
            for dim in x.shape[1:]:
                num_features *= dim
            x = x.view(batch_size, num_features)
        
        # 动态创建投影层（如果需要）
        if self.projection is None:
            input_dim = x.shape[-1]
            if input_dim != self.target_dim:
                self.projection = nn.Linear(input_dim, self.target_dim).to(x.device)
            else:
                self.projection = nn.Identity()
        
        # 应用投影层对齐维度
        x_aligned = self.projection(x)
        
        # 对于多维输入，仅保持批次维度，输出为 [batch_size, target_dim]
        # 不尝试恢复复杂的中间维度，避免reshape错误
        if len(original_shape) > 2:
            x_aligned = x_aligned.view(batch_size, self.target_dim)
        
        return x_aligned


        # 存放关键特征维度以便后续对齐
        self.target_fusion_dim = 768  # 标准化目标维度
        
        # 维度自动对齐器 - 所有认知模块输出64维对齐到768维
        self.ambient_dim_aligner = DimensionAligner(self.target_fusion_dim, module_output_dim)  # 环境流64→768维
        self.focal_dim_aligner = DimensionAligner(self.target_fusion_dim, module_output_dim)    # 焦点流64→768维  
        self.exif_dim_aligner = DimensionAligner(self.target_fusion_dim, module_output_dim)     # EXIF流64→768维
        self.feature_fusion_aligner = DimensionAligner(self.target_fusion_dim * 3, self.target_fusion_dim)  # 融合层2304→768维

        # 全局融合对齐器 - 解决不同类型特征最终融合时的维度统一
        self.global_aligner = DimensionAligner(self.target_fusion_dim * 3, self.target_fusion_dim)  # 2304→768维
        
        # 动态维度计算器 - 自动适配不同认知模块的输入输出
        self.dim_calculator = nn.ModuleDict({
            'ambient': nn.Sequential(nn.Linear(max(256, 192), 192), nn.ReLU()),
            'focal': nn.Sequential(nn.Linear(max(512, 192), 192), nn.ReLU()),  
            'exif': nn.Sequential(nn.Linear(max(32, 32), 32), nn.ReLU())
        })

def create_model(config, camera_info=None, device=None):
    """创建模型的工厂函数"""
    model = CognitiveAimModel(config, camera_info)
    
    # 将模型移动到指定设备
    if device is not None:
        model = model.to(device)
        # 确保 AdaptiveLoRAHead 的参数也在正确设备上
        if hasattr(model, 'decision_head') and hasattr(model.decision_head, 'to'):
            model.decision_head.to(device)
        if hasattr(model, 'confidence_head') and hasattr(model.confidence_head, 'to'):
            model.confidence_head.to(device)
        print(f"模型已移动到设备: {device}")
    
    # 加载预训练权重（如果指定）
    if config.get('load_checkpoint'):
        checkpoint_path = config['load_checkpoint']
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # 过滤掉可能不匹配的权重，包括认知模块权重
            filtered_state_dict = {}
            skip_prefixes = [
                'decision_head.', 'confidence_head.', 'curiosity_module.', 'global_aligner.',
                'ambient_stream.', 'focal_stream.', 'exif_prior.', 'fusion.'
            ]
            
            for key, value in state_dict.items():
                should_skip = any(key.startswith(prefix) for prefix in skip_prefixes)
                if not should_skip:
                    filtered_state_dict[key] = value
                else:
                    print(f"跳过加载权重: {key} (认知模块或头部权重)")
            
            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"成功加载预训练权重：{checkpoint_path}")
            print(f"决策头和置信度头将使用新的维度重新初始化")
            
            # 重新将模型移动到设备（以防权重加载后设备改变）
            if device is not None:
                model = model.to(device)
                
        except Exception as e:
            print(f"警告：无法加载预训练权重 {checkpoint_path}: {e}")
    
    # 最终确保所有参数都在正确设备上
    if device is not None:
        model = model.to(device)
        # 强制移动所有子模块
        for module in model.modules():
            module.to(device)
    
    return model


# 添加维度自动对齐器解决特征维度不匹配问题

