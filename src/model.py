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
        
        # Custom attention mechanism - avoid uniformization problem of MultiheadAttention
        self.query_proj = nn.Linear(patch_dim, patch_dim)
        self.key_proj = nn.Linear(patch_dim, patch_dim)
        self.value_proj = nn.Linear(patch_dim, patch_dim)
        self.scale = math.sqrt(patch_dim // num_heads)
        self.attention_dropout = nn.Dropout(0.0)
        
        # Curiosity-driven attention modulator
        if curiosity_guided:
            self.curiosity_modulator = nn.Sequential(
                nn.Linear(1, hidden_dim // 8),
                nn.ReLU(),
                nn.Linear(hidden_dim // 8, num_heads),
                nn.Sigmoid()
            )
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 4)
        )
        
        # Adaptive weight learning
        self.adaptive_weight = nn.Parameter(torch.tensor(0.5))
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to ensure focal stream has sufficient activation and attention diversity"""
        # Initialize projection layers
        for module in self.projection:
            if isinstance(module, nn.Linear):
                # Use more conservative Xavier initialization
                nn.init.xavier_uniform_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)  # Zero bias
        
        # Initialize curiosity modulator
        if self.curiosity_guided:
            for module in self.curiosity_modulator:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.8)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
        
        # Initialize custom attention weights to increase diversity
        with torch.no_grad():
            # Initialize query, key, value projection layers
            nn.init.xavier_normal_(self.query_proj.weight, gain=2.0)
            nn.init.xavier_normal_(self.key_proj.weight, gain=2.0)
            nn.init.xavier_normal_(self.value_proj.weight, gain=1.0)
            
            # Bias initialization
            if self.query_proj.bias is not None:
                nn.init.uniform_(self.query_proj.bias, -0.05, 0.05)
            if self.key_proj.bias is not None:
                nn.init.uniform_(self.key_proj.bias, -0.05, 0.05)
            if self.value_proj.bias is not None:
                nn.init.constant_(self.value_proj.bias, 0.0)
        
    def forward(self, patch_tokens, curiosity_score=None):
        """
        Args:
            patch_tokens: [batch_size, num_patches, patch_dim] - DINOv2 patch tokens
            curiosity_score: [batch_size] - Curiosity score (optional)
        Returns:
            focal_features: [batch_size, hidden_dim//4] - Focal features
            attention_weights: [batch_size, num_patches] - Attention weights
        """
        batch_size, num_patches, patch_dim = patch_tokens.size()
        
        # Add 2D position encoding
        def create_2d_position_encoding(num_patches, patch_dim, device):
            """Create 2D position encoding to help model understand spatial relationships"""
            # Ensure position encoding count exactly matches actual patch count
            pos_encoding = torch.zeros(num_patches, patch_dim, device=device)
            
            # Try 2D layout, use 1D if not perfect square
            patch_size = int(num_patches ** 0.5)
            if patch_size * patch_size == num_patches:
                # 2D position encoding (square layout)
                for i in range(num_patches):
                    row = i // patch_size
                    col = i % patch_size
                    
                    # Row position encoding (use first half of patch_dim)
                    if patch_dim >= 4:  # Ensure sufficient dimensions
                        div_term_row = torch.exp(torch.arange(0, patch_dim//2, 2, dtype=torch.float, device=device) * 
                                               -(math.log(10000.0) / (patch_dim//2)))
                        if len(div_term_row) > 0:
                            pos_encoding[i, 0:patch_dim//2:2] = torch.sin(row * div_term_row)
                            pos_encoding[i, 1:patch_dim//2:2] = torch.cos(row * div_term_row)
                        
                        # Column position encoding (use second half of patch_dim)
                        div_term_col = torch.exp(torch.arange(0, patch_dim//2, 2, dtype=torch.float, device=device) * 
                                               -(math.log(10000.0) / (patch_dim//2)))
                        if len(div_term_col) > 0:
                            pos_encoding[i, patch_dim//2::2] = torch.sin(col * div_term_col)
                            pos_encoding[i, patch_dim//2+1::2] = torch.cos(col * div_term_col)
            else:
                # 1D position encoding (non-square layout)
                position = torch.arange(0, num_patches, dtype=torch.float, device=device).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, patch_dim, 2, dtype=torch.float, device=device) * 
                                   -(math.log(10000.0) / patch_dim))
                if len(div_term) > 0:
                    pos_encoding[:, 0::2] = torch.sin(position * div_term)
                    if patch_dim > 1:
                        pos_encoding[:, 1::2] = torch.cos(position * div_term)
            
            return pos_encoding
        
        # Add position encoding to patch tokens
        try:
            pos_encoding = create_2d_position_encoding(num_patches, patch_dim, patch_tokens.device)
            # Ensure dimensions match exactly
            if pos_encoding.shape[0] == num_patches and pos_encoding.shape[1] == patch_dim:
                patch_tokens = patch_tokens + pos_encoding.unsqueeze(0)  # Broadcast to batch dimension
            else:
                print(f"Position encoding dimension mismatch, skipping: pos_encoding {pos_encoding.shape}, patch_tokens {patch_tokens.shape}")
        except Exception as e:
            print(f"Position encoding creation failed, skipping: {e}")
        
        # Custom attention computation
        # Compute queries, keys, values
        queries = self.query_proj(patch_tokens)  # [batch_size, num_patches, patch_dim]
        keys = self.key_proj(patch_tokens)       # [batch_size, num_patches, patch_dim]
        values = self.value_proj(patch_tokens)   # [batch_size, num_patches, patch_dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_patches, num_patches]
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention weights to values
        attended_patches = torch.matmul(attention_weights, values)  # [batch_size, num_patches, patch_dim]
        
        # Compute aggregated attention weights for each patch (for visualization and subsequent processing)
        # Use spatial weighted aggregation strategy, giving center regions higher base weights
        def create_center_bias_mask(num_patches, center_strength=0.3):
            """Create center bias mask to enhance attention weights in center regions"""
            # Calculate patch_size based on actual patch count
            patch_size = int(num_patches ** 0.5)
            if patch_size * patch_size != num_patches:
                # If not perfect square, use 1D center bias
                center_pos = num_patches // 2
                positions = torch.arange(num_patches, dtype=torch.float)
                distance = torch.abs(positions - center_pos)
                # Use more concentrated standard deviation
                sigma = num_patches / 12  # Changed from num_patches/8 to num_patches/12 for more concentrated distribution
                center_bias = torch.exp(-distance**2 / (2 * sigma**2))
                return center_bias * center_strength
            
            # 2D center bias (square layout)
            center = patch_size // 2
            y, x = torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size), indexing='ij')
            # Calculate distance from center
            distance = torch.sqrt((x - center).float()**2 + (y - center).float()**2)
            # Create more concentrated Gaussian distribution center bias (reduce standard deviation)
            sigma = patch_size / 6  # Changed from patch_size/4 to patch_size/6 for more concentrated distribution
            center_bias = torch.exp(-distance**2 / (2 * sigma**2))
            center_bias = center_bias.flatten()  # Flatten to 1D, length exactly equals num_patches
            return center_bias * center_strength
        
        # Use mean of attention weight matrix as base
        patch_attention = attention_weights.mean(dim=1)  # [batch_size, num_patches]
        
        # Add center bias
        num_patches = patch_attention.size(-1)
        center_bias = create_center_bias_mask(num_patches).to(patch_attention.device)
        patch_attention = patch_attention + center_bias.unsqueeze(0)  # Broadcast to batch dimension
        
        # If base weights are still uniform, use diagonal elements
        if patch_attention.var() < 1e-6:
            patch_attention = torch.diagonal(attention_weights, dim1=-2, dim2=-1)  # [batch_size, num_patches]
            patch_attention = patch_attention + center_bias.unsqueeze(0)
        
        # Final fallback: use maximum value of each row
        if patch_attention.var() < 1e-6:
            patch_attention, _ = torch.max(attention_weights, dim=-1)  # [batch_size, num_patches]
            patch_attention = patch_attention + center_bias.unsqueeze(0)
        
        # Final safety measure: weights based on input features (without softmax normalization)
        if patch_attention.var() < 1e-6:
            # Use L2 norm of input patch tokens as importance
            patch_norms = torch.norm(patch_tokens, dim=-1)  # [batch_size, num_patches]
            # Add noise to increase variation
            noise = torch.randn_like(patch_norms) * 0.1 * patch_norms.std()
            patch_attention = patch_norms + noise
        
        # Use gentler normalization to preserve more variation
        # Use simple L1 normalization instead of softmax
        patch_attention = patch_attention / (patch_attention.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Curiosity-driven attention modulation
        if self.curiosity_guided and curiosity_score is not None:
            # Convert curiosity score to attention modulation weights
            curiosity_modulation = self.curiosity_modulator(curiosity_score.unsqueeze(-1))  # [batch_size, num_heads]
            
            # Apply curiosity modulation to patch attention
            curiosity_weight = curiosity_modulation.mean(dim=-1, keepdim=True)  # [batch_size, 1]
            modulated_attention = patch_attention * (1.0 + curiosity_weight)  # [batch_size, num_patches]
            
            # Adaptive weight mixing of original attention and modulated attention
            final_attention = (self.adaptive_weight * modulated_attention + 
                             (1 - self.adaptive_weight) * patch_attention)
        else:
            final_attention = patch_attention
        
        # Preserve original distribution of attention weights, avoid over-smoothing by not using softmax
        # final_attention = F.softmax(final_attention, dim=-1)  # Commented out softmax
        # Ensure weights are positive and normalized
        final_attention = torch.clamp(final_attention, min=1e-8)
        final_attention = final_attention / (final_attention.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Attention entropy regularization - encourage attention weight concentration
        # Compute attention entropy (used during training, can be ignored during inference)
        if self.training:
            # Add small epsilon to avoid log(0)
            eps = 1e-8
            attention_entropy = -torch.sum(final_attention * torch.log(final_attention + eps), dim=-1)
            # Store entropy loss as instance attribute for training script access
            # Correction: use positive entropy as loss (encourage low entropy, i.e., more concentrated attention)
            entropy_loss = attention_entropy.mean()  # Positive entropy as loss (encourage concentrated attention)
            self._last_attention_entropy = entropy_loss
            
            # Debug info: check attention entropy calculation
            if hasattr(self, '_debug_entropy_count'):
                self._debug_entropy_count += 1
            else:
                self._debug_entropy_count = 1
            
            # Print debug info every 100 forward passes
            if self._debug_entropy_count % 100 == 0:
                print(f"[Debug] Attention entropy: {attention_entropy.mean().item():.6f}, Entropy loss: {entropy_loss.item():.6f}, Weight variance: {final_attention.var().item():.2e}")
        else:
            self._last_attention_entropy = 0.0
        
        # Weighted features
        weighted_features = torch.sum(attended_patches * final_attention.unsqueeze(-1), dim=1)
        
        # Feature projection
        focal_features = self.projection(weighted_features)
        
        return focal_features, final_attention

class IterativeFocalStream(nn.Module):
    """Iterative focal targeting stream - multi-step refined targeting (integrated curiosity-driven)"""
    
    def __init__(self, patch_dim, hidden_dim=256, num_iterations=2, curiosity_guided=True, focus_strength=0.1, attention_dropout=0.1):
        super().__init__()
        self.num_iterations = num_iterations
        self.curiosity_guided = curiosity_guided
        self.focus_strength = focus_strength  # Focus strength parameter
        
        # Multiple focal streams (integrated curiosity-driven) - pass attention_dropout parameter
        self.focal_streams = nn.ModuleList([
            FocalStream(patch_dim, hidden_dim, curiosity_guided=curiosity_guided, attention_dropout=attention_dropout) for _ in range(num_iterations)
        ])
        
        # Initial focus (for iterative focal stream)
        self.initial_focus = nn.Parameter(torch.randn(1, patch_dim))
        
        # Curiosity intensity regulator (for dynamic adjustment between iterations)
        if curiosity_guided:
            self.curiosity_amplifier = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, num_iterations),
                nn.Softmax(dim=-1)
            )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim // 4 * num_iterations, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to ensure iterative focal stream has sufficient activation and attention diversity"""
        # Initialize fusion layers
        for module in self.fusion:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # Initialize curiosity amplifier
        if self.curiosity_guided:
            for module in self.curiosity_amplifier:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.8)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
        
        # Initial focus parameter uses smaller standard deviation
        nn.init.normal_(self.initial_focus, mean=0.0, std=0.02)
        
        # Key fix: re-initialize attention weights for each FocalStream instance
        for i, focal_stream in enumerate(self.focal_streams):
            with torch.no_grad():
                # Use slightly different initialization for each iteration to increase diversity
                diversity_factor = 1.0 + 0.1 * i  # Increase diversity by 10% for each iteration
                
                # Re-initialize custom attention weights
                if hasattr(focal_stream, 'query_proj'):
                    nn.init.xavier_normal_(focal_stream.query_proj.weight, gain=1.2 * diversity_factor)
                    nn.init.xavier_normal_(focal_stream.key_proj.weight, gain=1.2 * diversity_factor)
                    nn.init.xavier_normal_(focal_stream.value_proj.weight, gain=1.0 * diversity_factor)
                    
                    # Bias initialization
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
            curiosity_score: [batch_size] - Curiosity score (optional)
        Returns:
            fused_features: [batch_size, hidden_dim//4]
            attention_weights: [batch_size, num_patches] - Attention weights from last iteration
        """
        features_list = []
        attention_weights = None
        
        current_patches = patch_tokens
        
        # Curiosity-driven iterative weight allocation
        if self.curiosity_guided and curiosity_score is not None:
            iteration_weights = self.curiosity_amplifier(curiosity_score.unsqueeze(-1))  # [batch_size, num_iterations]
        else:
            iteration_weights = None
        
        for i, focal_stream in enumerate(self.focal_streams):
            # Adjust curiosity intensity based on iteration position
            if iteration_weights is not None:
                # Use different curiosity intensity for each iteration
                iter_curiosity = curiosity_score * iteration_weights[:, i]
            else:
                iter_curiosity = curiosity_score
            
            features, attn_weights = focal_stream(current_patches, iter_curiosity)
            features_list.append(features)
            attention_weights = attn_weights  # Save attention weights from last iteration
            
            # Adjust patch tokens for next iteration based on curiosity attention weights
            if i < len(self.focal_streams) - 1:  # Not the last iteration
                # Re-focus patch tokens using attention weights
                enhanced_patches = current_patches * (1 + self.focus_strength * attn_weights.unsqueeze(-1))
                current_patches = enhanced_patches
        
        # Fuse features from all iterations
        fused_features = self.fusion(torch.cat(features_list, dim=1))
        
        # Collect attention entropy losses from all iterations
        if self.training:
            total_entropy_loss = 0.0
            entropy_count = 0
            for focal_stream in self.focal_streams:
                if hasattr(focal_stream, '_last_attention_entropy') and focal_stream._last_attention_entropy != 0:
                    total_entropy_loss += focal_stream._last_attention_entropy
                    entropy_count += 1
            
            # Calculate average entropy loss and store
            if entropy_count > 0:
                self._last_attention_entropy = total_entropy_loss / entropy_count
                # Debug info
                if not hasattr(self, '_debug_iter_count'):
                    self._debug_iter_count = 0
                self._debug_iter_count += 1
                if self._debug_iter_count % 50 == 0:
                    print(f"[Debug] IterativeFocalStream average entropy loss: {self._last_attention_entropy:.6f}, active streams: {entropy_count}/{len(self.focal_streams)}")
            else:
                self._last_attention_entropy = 0.0
        else:
            self._last_attention_entropy = 0.0
        
        return fused_features, attention_weights

class EXIFPriorDatabase(nn.Module):
    """EXIF experience database - encode camera parameters and experience (match checkpoint structure)"""
    
    def __init__(self, num_cameras, hidden_dim=256):
        super().__init__()
        self.num_cameras = num_cameras
        
        # Camera model embedding - match checkpoint
        self.camera_embedding = nn.Embedding(num_cameras, 64)
        
        # Continuous EXIF parameter encoder - match checkpoint structure
        self.exif_encoder = nn.Sequential(
            nn.Linear(3, 64),  # focal_length, aperture, iso
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Fusion layer - match checkpoint structure
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
        
        # Camera embedding
        camera_idx = exif_data['camera_idx']
        if camera_idx.dim() > 1:
            camera_idx = camera_idx.squeeze(1)
        
        camera_features = self.camera_embedding(camera_idx)
        
        # Process EXIF data
        focal_length = exif_data['focal_length']
        aperture = exif_data['aperture']
        iso = exif_data['iso']
        
        # Ensure correct dimensions
        if focal_length.dim() > 1:
            focal_length = focal_length.squeeze(1)
        if aperture.dim() > 1:
            aperture = aperture.squeeze(1)
        if iso.dim() > 1:
            iso = iso.squeeze(1)
        
        continuous_params = torch.stack([
            focal_length,
            aperture,
            torch.log(iso + 1)  # Log transform ISO
        ], dim=1)
        exif_features = self.exif_encoder(continuous_params)
        
        # Fusion
        combined_features = torch.cat([camera_features, exif_features], dim=1)
        return self.fusion(combined_features)

class CuriosityModule(nn.Module):
    """Variational Bayesian curiosity module - VAE-based uncertainty estimation"""
    
    def __init__(self, feature_dim, hidden_dim=128, enable_hierarchical=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.enable_hierarchical = enable_hierarchical
        self.latent_dim = feature_dim // 4  # Latent space dimension
        
        # Variational encoder
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
        
        # Variational decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, self.latent_dim)  # Reconstruct to compressed dimension
        )
        
        # Auxiliary uncertainty estimation network
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Softplus()  # Ensure positive output
        )
        
        # Hierarchical curiosity module
        if enable_hierarchical:
            # Geometric consistency curiosity (synergistic with EXIF)
            self.geometric_curiosity = nn.Sequential(
                nn.Linear(feature_dim + 4, hidden_dim),  # Features + EXIF geometric parameters
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            
            # Local exploration curiosity (detail discovery)
            self.local_curiosity = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            
            # Adaptive weight learning (geometric, local, variational Bayesian)
            self.curiosity_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.3]))
        
        # Exploration history record (for dynamic sampling)
        self.register_buffer('exploration_history', torch.zeros(1000))  # Circular buffer
        self.register_buffer('history_pointer', torch.tensor(0))
        
    def forward(self, features, targets=None, exif_data=None, loss_type="robust", uncertainty_weight=0.1):
        """
        Variational Bayesian curiosity mechanism: estimate cognitive uncertainty through VAE
        
        Args:
            features: [batch_size, feature_dim] - Model features
            targets: [batch_size] - Ground truth depth values (optional)
            exif_data: dict - EXIF information (for geometric curiosity)
            loss_type: str - Loss type ("simple", "robust", "huber")
            uncertainty_weight: float - Uncertainty weight
        Returns:
            curiosity_reward: [batch_size] - Comprehensive curiosity reward (non-negative)
            uncertainty_score: [batch_size] - Uncertainty score (non-negative)
            curiosity_components: dict - Component details
        """
        batch_size = features.size(0)
        #https://github.com/yenjane-dot/cognitive-aim-depth-estimation        
        # 1. Variational encoding: compute mean and variance of latent space
        mu = self.encoder_mean(features)  # [batch_size, latent_dim]
        logvar = self.encoder_logvar(features)  # [batch_size, latent_dim]
        
        # 2. Reparameterization trick: sample from latent distribution
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # [batch_size, latent_dim]
        
        # 3. Variational decoding: reconstruct features
        reconstructed = self.decoder(z)  # [batch_size, latent_dim]
        
        # 4. Compute reconstruction error (reconstruction term of variational lower bound)
        target_features = features[:, :reconstructed.size(1)].detach()  # Take corresponding dimensions
        
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
        
        # 5. Compute KL divergence (regularization term of variational lower bound)
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # 6. Auxiliary uncertainty estimation
        uncertainty_estimate = self.uncertainty_head(features).squeeze(-1)
        
        # 7. Comprehensive uncertainty score (ensure non-negative)
        reconstruction_error = torch.clamp(reconstruction_error, min=0.0)
        kl_divergence = torch.clamp(kl_divergence, min=0.0)
        uncertainty_estimate = torch.clamp(uncertainty_estimate, min=0.0, max=10.0)
        
        # Variational Bayesian uncertainty: reconstruction error + KL divergence + auxiliary estimation
        basic_uncertainty = reconstruction_error + 0.1 * kl_divergence + uncertainty_weight * uncertainty_estimate
        
        curiosity_components = {
            'reconstruction_error': reconstruction_error,
            'kl_divergence': kl_divergence,
            'uncertainty_estimate': uncertainty_estimate,
            'basic_uncertainty': basic_uncertainty,
            'latent_mean': mu,
            'latent_logvar': logvar
        }
        
        # 6. Hierarchical curiosity computation (all components ensure non-negative)
        if self.enable_hierarchical and hasattr(self, 'geometric_curiosity'):
            # Geometric consistency curiosity (based on EXIF prior)
            geometric_uncertainty = self._compute_geometric_curiosity(features, exif_data)
            
            # Local exploration curiosity (based on feature change sensitivity)
            local_uncertainty = self._compute_local_curiosity(features)
            
            # Weight normalization
            weights = F.softmax(self.curiosity_weights, dim=0)
            
            # Comprehensive curiosity reward (weighted average, ensure non-negative)
            curiosity_reward = (weights[0] * geometric_uncertainty + 
                              weights[1] * local_uncertainty + 
                              weights[2] * basic_uncertainty)
            
            curiosity_components.update({
                'geometric_uncertainty': geometric_uncertainty,
                'local_uncertainty': local_uncertainty,
                'weights': weights
            })
            
            # Update exploration history
            self._update_exploration_history(curiosity_reward.detach())
        else:
            curiosity_reward = basic_uncertainty
        
        # 7. Finally ensure all outputs are non-negative
        curiosity_reward = torch.clamp(curiosity_reward, min=0.0, max=100.0)
        uncertainty_score = torch.clamp(basic_uncertainty, min=0.0, max=100.0)
        
        return curiosity_reward, uncertainty_score, curiosity_components
    
    def _compute_geometric_curiosity(self, features, exif_data):
        """Compute geometric consistency uncertainty (Variational Bayesian version)
        
        Evaluate geometric uncertainty based on consistency between EXIF parameters and visual features.
        Combining variational inference ideas, quantify geometric understanding through uncertainty of feature-EXIF joint distribution.
        """
        batch_size = features.size(0)
        
        if exif_data is None:
            # Return medium uncertainty when no EXIF information available
            return torch.full((batch_size,), 0.5, device=features.device)
        
        try:
            # Combine EXIF geometric parameters (normalized processing)
            focal_length = exif_data.get('focal_length', torch.zeros(batch_size, device=features.device)).view(-1)
            aperture = exif_data.get('aperture', torch.zeros(batch_size, device=features.device)).view(-1)
            iso = exif_data.get('iso', torch.zeros(batch_size, device=features.device)).view(-1)
            
            # Normalize EXIF parameters to [0,1] range
            focal_length_norm = torch.clamp(focal_length / 200.0, 0.0, 1.0)  # Assume max focal length 200mm
            aperture_norm = torch.clamp(aperture / 32.0, 0.0, 1.0)  # Assume max aperture f/32
            iso_norm = torch.clamp(iso / 6400.0, 0.0, 1.0)  # Assume max ISO 6400
            
            exif_features = torch.stack([
                focal_length_norm,
                aperture_norm, 
                iso_norm,
                torch.ones(batch_size, device=features.device)  # Bias term
            ], dim=1)
            
            combined_features = torch.cat([features, exif_features], dim=1)
            geometric_uncertainty = self.geometric_curiosity(combined_features).squeeze(-1)
            
            # Sigmoid output is already in [0,1] range, representing uncertainty level
            return torch.clamp(geometric_uncertainty, min=0.0, max=1.0)
            
        except Exception as e:
            print(f"Geometric uncertainty computation failed: {e}")
            # Return high uncertainty when computation fails
            return torch.full((batch_size,), 0.8, device=features.device)
    
    def _compute_local_curiosity(self, features):
        """Compute local exploration uncertainty (Variational Bayesian version)
        
        Test model sensitivity to local changes through feature perturbation.
        Combining variational inference ideas, high sensitivity indicates uncertainty in latent space, reflecting instability of model understanding.
        """
        # Basic local uncertainty assessment
        local_uncertainty_base = self.local_curiosity(features).squeeze(-1)
        
        # Feature perturbation sensitivity test
        with torch.no_grad():
            # Add small random perturbation
            noise_scale = 0.01
            noise = torch.randn_like(features) * noise_scale
            noisy_features = features + noise
            
            # Compute uncertainty after perturbation
            noisy_uncertainty = self.local_curiosity(noisy_features).squeeze(-1)
            
            # Sensitivity = degree of difference before and after perturbation
            sensitivity = torch.abs(local_uncertainty_base - noisy_uncertainty)
            
        # Comprehensive local uncertainty: basic uncertainty + sensitivity weight
        # High sensitivity areas indicate unstable model understanding, should increase uncertainty
        local_uncertainty = local_uncertainty_base + sensitivity * 0.2
        
        # Ensure output is in reasonable range (Sigmoid + sensitivity adjustment)
        return torch.clamp(local_uncertainty, min=0.0, max=1.0)
    
    def _update_exploration_history(self, rewards):
        """Update exploration history"""
        if not hasattr(self, 'exploration_history') or self.exploration_history is None:
            return
        
        # Flatten rewards and update history
        if rewards.dim() > 1:
            rewards = rewards.mean(dim=-1)
        
        with torch.no_grad():
            for reward in rewards:
                idx = int(self.history_pointer) % self.exploration_history.size(0)
                self.exploration_history[idx] = reward.item()
                self.history_pointer = (self.history_pointer + 1) % self.exploration_history.size(0)
    
    def get_exploration_statistics(self):
        """Get exploration statistics"""
        if not hasattr(self, 'exploration_history') or self.exploration_history is None:
            return {'mean': 0., 'std': 0., 'max': 0., 'samples': 0}
        
        with torch.no_grad():
            # Filter out uninitialized values (assume initial is 0)
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
    """Cognitive-Aim main model"""
    
    def __init__(self, config, camera_info=None):
        super().__init__()
        self.config = config
        
        # DINOv2 backbone network
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
        
        # Freeze backbone network (optional)
        if config.get('freeze_backbone', True):
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # LoRA adaptation (optional)
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
        
        # Cognitive modules (support multi-level configuration reading)
        # Priority: read from model.cognitive_modules first, then from top-level cognitive_modules
        model_config = config.get('model', {})
        cognitive_modules = model_config.get('cognitive_modules', config.get('cognitive_modules', []))
        debug_init = config.get('debug_init', False)
        
        if debug_init:
            print(f"Cognitive modules in config: {cognitive_modules}")
            print(f"Read from path: {'model.cognitive_modules' if 'cognitive_modules' in model_config else 'cognitive_modules'}")
        
        # Global perception stream
        if 'ambient_stream' in cognitive_modules:
            self.ambient_stream = AmbientStream(self.feature_dim)
            self.use_ambient = True
            if debug_init:
                print("Global perception stream enabled")
        else:
            self.use_ambient = False
        
        # Focal aiming stream (support curiosity-driven)
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
                print(f"Iterative focal aiming stream enabled ({focal_config.get('num_iterations', 3)} iterations, dropout={attention_dropout})")
        elif 'focal_stream' in cognitive_modules:
            attention_dropout = config.get('curiosity_guided_attention', {}).get('attention_dropout', 0.1)
            self.focal_stream = FocalStream(self.feature_dim, attention_dropout=attention_dropout)
            self.use_focal = True
            self.use_iterative = False
            if debug_init:
                print(f"Basic focal aiming stream enabled (dropout={attention_dropout})")
        else:
            self.use_focal = False
            self.use_iterative = False
        
        # EXIF experience database
        if 'exif_prior_database' in cognitive_modules and camera_info:
            self.exif_prior = EXIFPriorDatabase(camera_info['num_cameras'])
            self.use_exif = True
            if debug_init:
                print("EXIF experience database enabled")
        else:
            self.use_exif = False
        
        # Calculate fusion feature dimension - match checkpoint structure
        # Note: actual output dimension of each cognitive module is hidden_dim // 4 = 256 // 4 = 64
        module_output_dim = 256 // 4  # 64 dimensions
        fusion_dim = 0
        if self.use_ambient:
            fusion_dim += module_output_dim  # AmbientStream output dimension is 64
        if self.use_focal:
            fusion_dim += module_output_dim  # FocalStream output dimension is 64
        if self.use_exif:
            fusion_dim += module_output_dim  # EXIFPriorDatabase output dimension is 64
        
        if fusion_dim == 0:
            fusion_dim = self.feature_dim  # If no cognitive modules, use CLS token directly
            
        # Fix: use fusion dimension 192 from checkpoint to match trained weights
        checkpoint_fusion_dim = 192  # Fusion layer dimension in checkpoint
        self.fusion_dim = checkpoint_fusion_dim
        
        # Fusion layer - match checkpoint dimensions
        self.fusion = nn.Sequential(
            nn.Linear(checkpoint_fusion_dim, checkpoint_fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Print fusion layer input dimension for debugging
        print(f"Fusion layer input dimension: {self.fusion_dim}")
        
        # Dimension aligner initialization - configure based on actual feature output dimensions
        self.target_fusion_dim = 768
        # All cognitive module outputs are 64-dimensional (hidden_dim//4)
        self.ambient_dim_aligner = DimensionAligner(target_dim=self.target_fusion_dim, source_dim=module_output_dim)
        self.focal_dim_aligner   = DimensionAligner(target_dim=self.target_fusion_dim, source_dim=module_output_dim)
        self.exif_dim_aligner    = DimensionAligner(target_dim=self.target_fusion_dim, source_dim=module_output_dim)

        # Decision head (match fusion layer output dimension 192) - use Softplus to ensure positive depth values
        self.decision_head = nn.Sequential(
            nn.Linear(checkpoint_fusion_dim, 1),
            nn.Softplus()  # Smooth positive activation function, avoid gradient vanishing
        )
        
        # Initialize decision head weights to ensure stable gradients and reasonable depth range
        with torch.no_grad():
            nn.init.xavier_uniform_(self.decision_head[0].weight, gain=1.0)
            nn.init.constant_(self.decision_head[0].bias, 1.0)  # Larger positive bias ensures reasonable depth range
        
        # Confidence head (match fusion layer output dimension 192)
        self.confidence_head = nn.Sequential(
            nn.Linear(checkpoint_fusion_dim, 1),
            nn.ReLU(),  # Use ReLU activation first
            nn.Linear(1, 1),
            nn.Sigmoid()  # Ensure confidence is in 0-1 range
        )
        
        # Initialize confidence head bias to tend to produce higher confidence
        with torch.no_grad():
            self.confidence_head[2].bias.fill_(2.0)  # Add positive bias
        
        # Enhanced curiosity module (restore original configuration to match checkpoint)
        self.curiosity_module = CuriosityModule(
            self.target_fusion_dim, 
            hidden_dim=256,  # Restore original hidden layer dimension
            enable_hierarchical=config.get('enable_hierarchical_curiosity', True)
        )
        print("[Debug] Curiosity module enabled (hidden dimension 256, matching checkpoint)")
        
        # Global fusion aligner - dynamically adapt concatenated feature dimensions
        # If all 3 modules enabled: 3*768=2304 dimensions, if 2 modules: 2*768=1536 dimensions
        max_source_dim = self.target_fusion_dim * 3  # Maximum possible source dimension
        self.global_aligner = DimensionAligner(target_dim=self.target_fusion_dim, source_dim=max_source_dim)
        
    def get_features_aligned(self, images, exif_data=None):
        """Extract features and implement automatic dimension alignment for cognitive architecture internal use
        
        Solve 192×192 vs 32×128 matrix multiplication dimension mismatch problem
        
        Args:
            images: [batch_size, 3, H, W]
            exif_data: dict with EXIF information
        Returns:
            fused_features: [batch_size, fusion_dim] - Features after dimension unification
        """
        try:
            # Extract DINOv2 features
            outputs = self.backbone(images, output_hidden_states=True)
            cls_token = outputs.last_hidden_state[:, 0]  # [B, hidden_dim]
            patch_tokens = outputs.last_hidden_state[:, 1:]  # [B, N, hidden_dim]
        
            # Cognitive module processing
            cognitive_features = []
            feature_dims = []
            
            # 1. Global perception stream (CLS token → 64 dimensions aligned to 768 dimensions)
            if self.use_ambient:
                ambient_raw = self.ambient_stream(cls_token)
                ambient_aligned = self.ambient_dim_aligner(ambient_raw)
                cognitive_features.append(ambient_aligned)
                feature_dims.append(768)
        
            # 2. Focal aiming stream (patch tokens → 64 dimensions aligned to 768 dimensions)
            if self.use_focal:
                # Calculate preliminary curiosity score (based on CLS token)
                curiosity_score = None
                if hasattr(self, 'curiosity_module'):
                    try:
                        cls_features = cls_token.view(cls_token.size(0), -1)
                        initial_curiosity, _, _ = self.curiosity_module(cls_features)
                        curiosity_score = initial_curiosity.mean(dim=-1) if initial_curiosity.dim() > 1 else initial_curiosity
                    except Exception as e:
                        print(f"Preliminary curiosity calculation failed, using default value: {e}")
                        # Provide default curiosity score
                        batch_size = cls_token.size(0)
                        curiosity_score = torch.ones(batch_size, device=cls_token.device) * 0.5
                else:
                    # If no curiosity module, provide default score
                    batch_size = cls_token.size(0)
                    curiosity_score = torch.ones(batch_size, device=cls_token.device) * 0.5
                
                # Uniformly use focal_stream, whether IterativeFocalStream or FocalStream is assigned to this attribute
                focal_raw, attn_weights = self.focal_stream(patch_tokens, curiosity_score)
                
                focal_aligned = self.focal_dim_aligner(focal_raw)
                cognitive_features.append(focal_aligned)
                feature_dims.append(768)
        
            # Directly get original 64-dimensional features to match checkpoint structure
            # Skip 768-dimensional alignment logic, use 64-dimensional features directly
            raw_features = []
            if self.use_ambient:
                ambient_raw = self.ambient_stream(cls_token)  # 64 dimensions
                raw_features.append(ambient_raw)
            if self.use_focal:
                focal_raw, _ = self.focal_stream(patch_tokens, curiosity_score)  # 64 dimensions
                raw_features.append(focal_raw)
            if self.use_exif and exif_data is not None:
                exif_raw = self.exif_prior(exif_data)  # 64 dimensions
                raw_features.append(exif_raw)
            
            # Handle feature dimension mismatch
            if len(raw_features) == 0:
                raise RuntimeError("Cognitive modules returned no features!")
            
            # Concatenate 64-dimensional features, pad with zeros if EXIF module is missing to match checkpoint's 192 dimensions
            concatenated_features = torch.cat(raw_features, dim=1)  # [B, 64*n]
            
            # If feature dimension is less than 192 (missing EXIF module), pad with zeros
            if concatenated_features.size(1) < 192:
                batch_size = concatenated_features.size(0)
                padding_size = 192 - concatenated_features.size(1)
                padding = torch.zeros(batch_size, padding_size, device=concatenated_features.device)
                concatenated_features = torch.cat([concatenated_features, padding], dim=1)
                print(f"[Debug] Missing EXIF module, padding with zeros to 192 dimensions")
            
            print(f"[Debug] Concatenated feature dimension: {concatenated_features.shape}")
            
            # Process through fusion layer to match checkpoint structure
            fused_features = self.fusion(concatenated_features)
            print(f"[Debug] Fused feature dimension: {fused_features.shape}")
            
            return fused_features
        
        except Exception as e:
            print(f"get_features_aligned exception: {e}")
            traceback.print_exc()
            # Return backup features
            backup_features = torch.zeros(images.size(0), self.fusion_dim, device=images.device)
            print(f"Return backup feature dimension: {backup_features.shape}")
            return backup_features
    
    def get_attention_weights(self):
        """Get attention weights from last forward propagation"""
        if hasattr(self, '_last_attention_weights'):
            return self._last_attention_weights
        return None

    def forward(self, images, exif_data=None, return_attention=False):
        """Forward propagation function - integrated cognitive architecture full pipeline with dimension alignment
        
        Args:
            images: [batch_size, 3, H, W] - Input images
            exif_data: dict with EXIF information  
            return_attention: bool - Whether to return attention weights
            
        Returns:
            depth_pred: [batch_size, 1] - Depth prediction
            confidence: [batch_size, 1] - Confidence
            attention_weights: (optional) Attention weights
        """
        # Use new aligned feature extraction method
        aligned_features = self.get_features_aligned(images, exif_data)
        
        if aligned_features is None:
            raise ValueError(f"Feature alignment result abnormal: None")
        
        # Feature dimension check
        if aligned_features.shape[1] != 192:
            raise ValueError(f"Feature dimension mismatch, expected 192 dimensions, actual {aligned_features.shape[1]} dimensions")

        # Uniformly apply aligned features for prediction
        fused_features = aligned_features  # Already aligned
        self.fusion_features = fused_features  # Store fused features for curiosity module use
        
        # Get and save attention weights (for visualization)
        # Only set basic weights when no saved guided attention weights exist
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
        
        # Depth prediction (using standardized feature dimensions)
        depth_pred = self.decision_head(fused_features)
        
        # Confidence prediction (also using aligned features)
        confidence = self.confidence_head(fused_features)
        
        # Unified return result format
        if return_attention:
            if self.use_focal:
                # Getting attention weights requires re-using original forward
                outputs = self.backbone(images, output_hidden_states=True)
                cls_token = outputs.last_hidden_state[:, 0]
                patch_tokens = outputs.last_hidden_state[:, 1:]
                
                # Use actual curiosity score instead of hardcoded value
                batch_size = patch_tokens.size(0)
                device = patch_tokens.device
                if hasattr(self, 'curiosity_module'):
                    try:
                        cls_features = cls_token.view(cls_token.size(0), -1)
                        initial_curiosity, _, _ = self.curiosity_module(cls_features)
                        curiosity_score = initial_curiosity.mean(dim=-1) if initial_curiosity.dim() > 1 else initial_curiosity
                        # Ensure curiosity score is in reasonable range
                        curiosity_score = torch.clamp(curiosity_score, 0.5, 1.0)
                    except Exception as e:
                        print(f"Curiosity module calculation failed, using maximum value: {e}")
                        curiosity_score = torch.ones(batch_size, device=device) * 1.0
                else:
                    curiosity_score = torch.ones(batch_size, device=device) * 1.0
                
                _, attn_weights = self.focal_stream(patch_tokens, curiosity_score)
                return depth_pred, confidence, attn_weights
            else:
                # If no focal stream, return None as attention weights
                return depth_pred, confidence, None
        
        return depth_pred, confidence
    
    def forward_with_guidance(self, images, exif_data=None, attention_guidance=None, return_attention=False):
        """Forward propagation function with user instruction guidance support
        
        Args:
            images: [batch_size, 3, H, W] - Input images
            exif_data: dict with EXIF information
            attention_guidance: [196] - User instruction generated attention guidance weights
            return_attention: bool - Whether to return attention weights
            
        Returns:
            depth_pred: [batch_size, 1] - Depth prediction
            confidence: [batch_size, 1] - Confidence
            attention_weights: (optional) Attention weights
        """
        try:
            # Extract DINOv2 features
            outputs = self.backbone(images, output_hidden_states=True)
            cls_token = outputs.last_hidden_state[:, 0]  # [B, hidden_dim]
            patch_tokens = outputs.last_hidden_state[:, 1:]  # [B, N, hidden_dim]
        
            # Cognitive module processing - use same 192-dimensional fusion logic as forward method
            attention_weights = None
            
            # Calculate preliminary curiosity score (based on CLS token)
            curiosity_score = None
            if hasattr(self, 'curiosity_module'):
                try:
                    cls_features = cls_token.view(cls_token.size(0), -1)
                    initial_curiosity, _, _ = self.curiosity_module(cls_features)
                    curiosity_score = initial_curiosity.mean(dim=-1) if initial_curiosity.dim() > 1 else initial_curiosity
                except Exception as e:
                    print(f"Preliminary curiosity calculation failed, using default value: {e}")
                    curiosity_score = None
            
            # Get original 64-dimensional features to match checkpoint structure
            raw_features = []
            
            # 1. Global perception stream (CLS token → 64 dimensions)
            if self.use_ambient:
                ambient_raw = self.ambient_stream(cls_token)  # 64 dimensions
                raw_features.append(ambient_raw)
        
            # 2. Focal aiming stream (patch tokens → 64 dimensions) - apply user guidance
            if self.use_focal:
                # Apply user instruction guided focal aiming
                if attention_guidance is not None:
                    focal_raw, attn_weights = self._guided_focal_stream(
                        patch_tokens, curiosity_score, attention_guidance
                    )
                else:
                    focal_raw, attn_weights = self.focal_stream(patch_tokens, curiosity_score)
                
                raw_features.append(focal_raw)  # 64 dimensions
                attention_weights = attn_weights
                # Save attention weights for visualization use
                self._last_attention_weights = attn_weights
        
            # 3. EXIF experience database (64 dimensions)  
            if self.use_exif and exif_data is not None:
                exif_raw = self.exif_prior(exif_data)  # 64 dimensions
                raw_features.append(exif_raw)
        
            # Concatenate 64-dimensional features: 3 modules = 192 dimensions, match checkpoint
            if len(raw_features) == 0:
                raise RuntimeError("Cognitive modules returned no features!")
            
            concatenated_features = torch.cat(raw_features, dim=1)  # [B, 64*n]
            
            # Process through fusion layer to match checkpoint structure
            fused_features = self.fusion(concatenated_features)
            
            # Directly use fused features for prediction
            depth_pred = self.decision_head(fused_features)
            confidence = self.confidence_head(fused_features)
            
            if return_attention:
                return depth_pred, confidence, attention_weights
            else:
                return depth_pred, confidence
            
        except Exception as e:
            print(f"Guided feature extraction failed: {e}")
            # Fall back to standard forward propagation
            return self.forward(images, exif_data, return_attention)
    
    def _guided_focal_stream(self, patch_tokens, curiosity_score, attention_guidance):
        """Apply user instruction guided focal aiming stream
        
        Args:
            patch_tokens: [batch_size, num_patches, patch_dim]
            curiosity_score: [batch_size] - Curiosity score
            attention_guidance: [num_patches] - User instruction guidance weights
            
        Returns:
            focal_features: [batch_size, hidden_dim//4]
            attention_weights: [batch_size, num_patches]
        """
        batch_size = patch_tokens.size(0)
        
        # Get basic attention weights
        base_features, base_attention = self.focal_stream(patch_tokens, curiosity_score)
        
        # Apply user instruction guidance
        if attention_guidance is not None:
            # Handle string type guidance instructions
            if isinstance(attention_guidance, str):
                # Convert string instructions to spatial attention weights
                num_patches = patch_tokens.size(1)
                patch_size = int(math.sqrt(num_patches))  # Assume square patch layout
                
                # Create spatial attention mask
                spatial_mask = torch.ones(patch_size, patch_size, device=patch_tokens.device)
                
                if attention_guidance.lower() == 'center':
                    # Center region enhancement
                    center_y, center_x = patch_size // 2, patch_size // 2
                    radius = max(1, patch_size // 4)
                    for y in range(patch_size):
                        for x in range(patch_size):
                            dist = math.sqrt((y - center_y)**2 + (x - center_x)**2)
                            if dist <= radius:
                                spatial_mask[y, x] = 3.0  # Center region weight enhancement
                            elif dist <= radius * 2:
                                spatial_mask[y, x] = 1.5  # Medium weight
                                
                elif attention_guidance.lower() == 'left':
                    # Left focus point (using circular focus pattern)
                    focus_y, focus_x = patch_size // 2, patch_size // 4  # Left 1/4 position
                    radius = max(1, patch_size // 6)  # Smaller focus radius
                    for y in range(patch_size):
                        for x in range(patch_size):
                            dist = math.sqrt((y - focus_y)**2 + (x - focus_x)**2)
                            if dist <= radius:
                                spatial_mask[y, x] = 5.0  # Strong focus
                            elif dist <= radius * 2:
                                spatial_mask[y, x] = 2.0  # Medium weight
                    
                elif attention_guidance.lower() == 'right':
                    # Right focus point (using circular focus pattern)
                    focus_y, focus_x = patch_size // 2, patch_size * 3 // 4  # Right 3/4 position
                    radius = max(1, patch_size // 6)  # Smaller focus radius
                    for y in range(patch_size):
                        for x in range(patch_size):
                            dist = math.sqrt((y - focus_y)**2 + (x - focus_x)**2)
                            if dist <= radius:
                                spatial_mask[y, x] = 5.0  # Strong focus
                            elif dist <= radius * 2:
                                spatial_mask[y, x] = 2.0  # Medium weight
                    
                elif attention_guidance.lower() == 'top':
                    # Top focus point (using circular focus pattern)
                    focus_y, focus_x = patch_size // 4, patch_size // 2  # Top 1/4 position
                    radius = max(1, patch_size // 6)  # Smaller focus radius
                    for y in range(patch_size):
                        for x in range(patch_size):
                            dist = math.sqrt((y - focus_y)**2 + (x - focus_x)**2)
                            if dist <= radius:
                                spatial_mask[y, x] = 5.0  # Strong focus
                            elif dist <= radius * 2:
                                spatial_mask[y, x] = 2.0  # Medium weight
                    
                elif attention_guidance.lower() == 'bottom':
                    # Bottom focus point (using circular focus pattern)
                    focus_y, focus_x = patch_size * 3 // 4, patch_size // 2  # Bottom 3/4 position
                    radius = max(1, patch_size // 6)  # Smaller focus radius
                    for y in range(patch_size):
                        for x in range(patch_size):
                            dist = math.sqrt((y - focus_y)**2 + (x - focus_x)**2)
                            if dist <= radius:
                                spatial_mask[y, x] = 5.0  # Strong focus
                            elif dist <= radius * 2:
                                spatial_mask[y, x] = 2.0  # Medium weight
                                
                elif attention_guidance.lower() in ['top-left', 'topleft']:
                    # Top-left corner focus point
                    focus_y, focus_x = patch_size // 4, patch_size // 4  # Top-left 1/4 position
                    radius = max(1, patch_size // 6)  # Smaller focus radius
                    for y in range(patch_size):
                        for x in range(patch_size):
                            dist = math.sqrt((y - focus_y)**2 + (x - focus_x)**2)
                            if dist <= radius:
                                spatial_mask[y, x] = 5.0  # Strong focus
                            elif dist <= radius * 2:
                                spatial_mask[y, x] = 2.0  # Medium weight
                                
                elif attention_guidance.lower() in ['top-right', 'topright']:
                    # Top-right corner focus point
                    focus_y, focus_x = patch_size // 4, patch_size * 3 // 4  # Top-right position
                    radius = max(1, patch_size // 6)  # Smaller focus radius
                    for y in range(patch_size):
                        for x in range(patch_size):
                            dist = math.sqrt((y - focus_y)**2 + (x - focus_x)**2)
                            if dist <= radius:
                                spatial_mask[y, x] = 5.0  # Strong focus
                            elif dist <= radius * 2:
                                spatial_mask[y, x] = 2.0  # Medium weight
                                
                elif attention_guidance.lower() in ['bottom-left', 'bottomleft']:
                    # Bottom-left corner focus point
                    focus_y, focus_x = patch_size * 3 // 4, patch_size // 4  # Bottom-left position
                    radius = max(1, patch_size // 6)  # Smaller focus radius
                    for y in range(patch_size):
                        for x in range(patch_size):
                            dist = math.sqrt((y - focus_y)**2 + (x - focus_x)**2)
                            if dist <= radius:
                                spatial_mask[y, x] = 5.0  # Strong focus
                            elif dist <= radius * 2:
                                spatial_mask[y, x] = 2.0  # Medium weight
                                
                elif attention_guidance.lower() in ['bottom-right', 'bottomright']:
                    # Bottom-right corner focus point
                    focus_y, focus_x = patch_size * 3 // 4, patch_size * 3 // 4  # Bottom-right position
                    radius = max(1, patch_size // 6)  # Smaller focus radius
                    for y in range(patch_size):
                        for x in range(patch_size):
                            dist = math.sqrt((y - focus_y)**2 + (x - focus_x)**2)
                            if dist <= radius:
                                spatial_mask[y, x] = 5.0  # Strong focus
                            elif dist <= radius * 2:
                                spatial_mask[y, x] = 2.0  # Medium weight
                
                # Flatten 2D mask to 1D
                attention_guidance = spatial_mask.flatten()
                
                # Continue processing converted numerical guidance weights
                pass
            
            # Ensure guidance weights match patch count
            num_patches = patch_tokens.size(1)
            if attention_guidance.size(0) != num_patches:
                # If dimensions don't match, perform interpolation adjustment
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
            
            # Expand to batch dimension
            guidance_batch = attention_guidance.unsqueeze(0).expand(batch_size, -1)
            
            # Use weighted fusion instead of simple multiplication to maintain stronger guidance effect
            alpha = 0.7  # Guidance weight influence strength
            guided_attention = alpha * guidance_batch + (1 - alpha) * base_attention
            
            # Use temperature-scaled softmax to maintain more contrast
            temperature = 0.05  # Lower temperature to enhance contrast
            guided_attention = F.softmax(guided_attention / temperature, dim=-1)
            
            # Recalculate features using guided attention
            weighted_patches = torch.sum(
                patch_tokens * guided_attention.unsqueeze(-1), dim=1
            )
            
            # Feature projection - ensure 64-dimensional output to match other cognitive modules
            if hasattr(self.focal_stream, 'projection'):
                guided_features = self.focal_stream.projection(weighted_patches)
            else:
                # If no projection layer, create a temporary linear layer with 64-dimensional output
                temp_projection = nn.Linear(self.feature_dim, 64).to(patch_tokens.device)
                guided_features = temp_projection(weighted_patches)
            
            return guided_features, guided_attention
        else:
            return base_features, base_attention
        
    def get_features(self, images, exif_data=None):
        """Maintain backward compatible feature extraction interface"""
        return self.get_features_aligned(images, exif_data)

    def compute_curiosity_loss(self, features, depth_targets=None, exif_data=None, loss_type="robust", uncertainty_weight=0.1):
        """Compute enhanced curiosity loss
        
        Args:
            features: [batch_size, target_fusion_dim] - Fused features
            depth_targets: [batch_size] - Target depth values (optional)
            exif_data: dict - EXIF information (for geometric curiosity)
            loss_type: str - Loss type
            uncertainty_weight: float - Uncertainty weight
            
        Returns:
            tuple: (curiosity_reward, curiosity_components)
        """
        try:
            curiosity_reward, prediction_error, curiosity_components = self.curiosity_module(
                features, depth_targets, exif_data, loss_type, uncertainty_weight
            )
            return curiosity_reward, curiosity_components
            
        except Exception as e:
            print(f"Enhanced curiosity loss calculation failed: {e}")
            batch_size = features.size(0)
            return torch.zeros(batch_size, device=features.device), {}

    def get_exploration_stats(self):
        """Get exploration statistics"""
        if hasattr(self, 'curiosity_module'):
            return self.curiosity_module.get_exploration_statistics()
        return {'mean': 0., 'std': 0., 'max': 0., 'samples': 0}


# End of file - duplicate create_model function removed
class DimensionAligner(nn.Module):
    """Automatically align feature vectors of different dimensions"""
    
    def __init__(self, target_dim, source_dim=None):
        super().__init__()
        self.target_dim = target_dim
        self.source_dim = source_dim
        
        # If source dimension is not specified, use adaptive projection
        if source_dim is None:
            self.projection = None  # Lazy initialization
        else:
            self.projection = nn.Linear(source_dim, target_dim) if target_dim != source_dim else nn.Identity()
        
    def forward(self, x):
        """
        Args:
            x: Input features [batch_size, ..., source_dim] or [batch_size, ..., target_dim]
        Returns:
            Aligned features [batch_size, ..., target_dim]
        """
        original_shape = x.shape
        batch_size = x.shape[0]
        
        # Flatten all dimensions except batch dimension to 2D [batch_size, features]
        if len(x.shape) > 2:
            # Calculate total number of features excluding batch dimension
            num_features = 1
            for dim in x.shape[1:]:
                num_features *= dim
            x = x.view(batch_size, num_features)
        
        # Dynamically create projection layer (if needed)
        if self.projection is None:
            input_dim = x.shape[-1]
            if input_dim != self.target_dim:
                self.projection = nn.Linear(input_dim, self.target_dim).to(x.device)
            else:
                self.projection = nn.Identity()
        
        # Apply projection layer to align dimensions
        x_aligned = self.projection(x)
        
        # For multi-dimensional input, only keep batch dimension, output as [batch_size, target_dim]
        # Don't attempt to restore complex intermediate dimensions to avoid reshape errors
        if len(original_shape) > 2:
            x_aligned = x_aligned.view(batch_size, self.target_dim)
        
        return x_aligned


        # Store key feature dimensions for subsequent alignment
        self.target_fusion_dim = 768  # Standardized target dimension
        
        # Automatic dimension aligners - all cognitive module outputs 64-dim aligned to 768-dim
        self.ambient_dim_aligner = DimensionAligner(self.target_fusion_dim, module_output_dim)  # Ambient stream 64→768 dim
        self.focal_dim_aligner = DimensionAligner(self.target_fusion_dim, module_output_dim)    # Focal stream 64→768 dim  
        self.exif_dim_aligner = DimensionAligner(self.target_fusion_dim, module_output_dim)     # EXIF stream 64→768 dim
        self.feature_fusion_aligner = DimensionAligner(self.target_fusion_dim * 3, self.target_fusion_dim)  # Fusion layer 2304→768 dim

        # Global fusion aligner - solve dimension unification when different types of features are finally fused
        self.global_aligner = DimensionAligner(self.target_fusion_dim * 3, self.target_fusion_dim)  # 2304→768 dim
        
        # Dynamic dimension calculator - automatically adapt input/output of different cognitive modules
        self.dim_calculator = nn.ModuleDict({
            'ambient': nn.Sequential(nn.Linear(max(256, 192), 192), nn.ReLU()),
            'focal': nn.Sequential(nn.Linear(max(512, 192), 192), nn.ReLU()),  
            'exif': nn.Sequential(nn.Linear(max(32, 32), 32), nn.ReLU())
        })

def create_model(config, camera_info=None, device=None):
    """Factory function for creating models"""
    model = CognitiveAimModel(config, camera_info)
    
    # Move model to specified device
    if device is not None:
        model = model.to(device)
        # Ensure AdaptiveLoRAHead parameters are also on the correct device
        if hasattr(model, 'decision_head') and hasattr(model.decision_head, 'to'):
            model.decision_head.to(device)
        if hasattr(model, 'confidence_head') and hasattr(model.confidence_head, 'to'):
            model.confidence_head.to(device)
        print(f"Model moved to device: {device}")
    
    # Load pretrained weights (if specified)
    if config.get('load_checkpoint'):
        checkpoint_path = config['load_checkpoint']
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # Filter out potentially mismatched weights, including cognitive module weights
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
                    print(f"Skip loading weight: {key} (cognitive module or head weight)")
            
            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"Successfully loaded pretrained weights: {checkpoint_path}")
            print(f"Decision head and confidence head will be reinitialized with new dimensions")
            
            # Move model back to device (in case device changes after weight loading)
            if device is not None:
                model = model.to(device)
                
        except Exception as e:
            print(f"Warning: Unable to load pretrained weights {checkpoint_path}: {e}")
    
    # Finally ensure all parameters are on the correct device
    if device is not None:
        model = model.to(device)
        # Force move all submodules
        for module in model.modules():
            module.to(device)
    
    return model
    

