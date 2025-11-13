import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchaudio
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from transformers import Wav2Vec2Model
import warnings
from torchvision.transforms.functional import center_crop

# This print statement MUST appear when you run the script.
print("--- [DEBUG] v4 model.py with vectorized GraphSwin is loaded. ---")

# --- 1. Raw Audio Stream (Wav2Vec 2.0 + Conformer) ---

class ConformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_dim, kernel_size=3, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.conv_block = nn.Sequential(
            # 1. Pointwise Conv (expand)
            nn.Conv1d(embed_dim, ff_dim, kernel_size=1),
            nn.SiLU(),
            # 2. Depthwise Conv (ff_dim channels)
            nn.Conv1d(ff_dim, ff_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=ff_dim),
            # 3. BatchNorm (on ff_dim)
            nn.BatchNorm1d(ff_dim),
            # 4. Activation
            nn.SiLU(),
            # 5. Pointwise Conv (contract)
            nn.Conv1d(ff_dim, embed_dim, kernel_size=1),
            # 6. Dropout
            nn.Dropout(dropout)
        )
        
        self.ff_block = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attention_mask=None):
        # x shape: (batch_size, seq_len, embed_dim)
        
        # 1. Feed-Forward
        x_ff = self.ff_block(x)
        x = x + 0.5 * F.dropout(x_ff, self.training)
        x = self.norm1(x)

        # 2. Multi-Head Self-Attention
        attn_out, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        x = x + F.dropout(attn_out, self.training)
        x = self.norm2(x)

        # 3. Convolution
        x_conv = x.transpose(1, 2)
        x_conv = self.conv_block(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x = x + F.dropout(x_conv, self.training)
        x = self.norm3(x)
        
        return x

class RawWav2VecConformer(nn.Module):
    def __init__(self, out_dim=256, n_conformers=2, n_heads=4, freeze_w2v=True):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        
        if freeze_w2v:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
            
        w2v_embed_dim = self.wav2vec2.config.hidden_size # 768
        
        self.conformer_blocks = nn.ModuleList(
            [ConformerBlock(w2v_embed_dim, n_heads, ff_dim=w2v_embed_dim*2) for _ in range(n_conformers)]
        )
        
        self.attention_pool = nn.Sequential(
            nn.Linear(w2v_embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Linear(w2v_embed_dim, out_dim)

    def forward(self, waveforms_padded, attention_mask):
        outputs = self.wav2vec2(waveforms_padded, attention_mask=attention_mask)
        x = outputs.last_hidden_state # (batch, seq_len, 768)
        
        original_lengths = attention_mask.sum(dim=1)
        output_lengths = self.wav2vec2._get_feat_extract_output_lengths(original_lengths)
        
        seq_len = x.shape[1]
        conformer_mask = torch.arange(seq_len, device=x.device).expand(x.shape[0], seq_len) >= output_lengths.unsqueeze(1)
        
        for block in self.conformer_blocks:
            x = block(x, attention_mask=conformer_mask)
        
        w = self.attention_pool(x)
        w = w.masked_fill(conformer_mask.unsqueeze(-1), -1e9) 
        w = F.softmax(w, dim=1)
        
        x = torch.sum(x * w, dim=1)
        
        x = F.relu(self.fc(x))
        return x

# --- 2. Spectrogram (Graph-Swin) Stream ---

class GraphSwinEncoder(nn.Module):
    def __init__(self, out_dim=256, gat_heads=4, n_mels=128):
        super().__init__()
        self.n_mels = n_mels
        
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            in_chans=3,     
            num_classes=0
        )
        
        # --- FIX: Freeze the Swin Transformer to prevent overfitting ---
        # This stops the 28M parameters from training
        for param in self.swin.parameters():
            param.requires_grad = False
        # --- End of Fix ---

        swin_embed_dim = 768
        
        self.gat1 = GATv2Conv(swin_embed_dim, 128, heads=gat_heads, concat=True, dropout=0.1)
        self.gat2 = GATv2Conv(128 * gat_heads, out_dim, heads=1, concat=False, dropout=0.1)
        self.grid_edges = self.create_grid_edges(7, 7)

    def forward(self, spec_tensors):
        # spec_tensors: (batch_size, n_mels, n_frames) (e.g., 16, 128, 500)
        
        # --- START: VECTORIZED BATCH PROCESSING (Replaces the loop) ---
        batch_size, n_mels, n_frames = spec_tensors.shape
        
        # (B, 1, 128, 500)
        spec_1ch = spec_tensors.unsqueeze(1)
        
        # Pad if smaller than 224x224
        pad_h = max(0, 224 - n_mels)
        pad_w = max(0, 224 - n_frames)
        spec_1ch_padded = F.pad(spec_1ch, (0, pad_w, 0, pad_h), 'constant', 0)
        
        # (B, 1, 224, 224)
        spec_1ch_cropped = center_crop(spec_1ch_padded, (224, 224))
        
        # (B, 3, 224, 224)
        spec_3ch = spec_1ch_cropped.repeat(1, 3, 1, 1)
        
        # (B, 49, 768)
        x_patches = self.swin.forward_features(spec_3ch) 
        
        # --- FIX: Handle 4D (B, H, W, C) or 3D (B, L, C) output from Swin ---
        if x_patches.dim() == 4:
            # Input is (B, H, W, C), e.g., (16, 7, 7, 768)
            B, H, W, C = x_patches.shape
            x_patches = x_patches.view(B, H * W, C) # Reshape to (16, 49, 768)
        
        # If input was (B, L, C), e.g., (16, 49, 768), dim is 3 and we skip the 'if'
        # Now x_patches is guaranteed to be (B, 49, 768)
        
        # --- END: VECTORIZED BATCH PROCESSING ---
        
        graph_data_list = []
        edge_index = self.grid_edges.to(spec_tensors.device)
        
        for i in range(batch_size):
            patches = x_patches[i] # (49, 768)
            graph_data_list.append(Data(x=patches, edge_index=edge_index))

        # This creates the batch and makes x 2D: (B * 49, 768)
        graph_batch = Batch.from_data_list(graph_data_list)
        
        x, edge_index = graph_batch.x, graph_batch.edge_index
        
        # This assertion should now pass
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.gat2(x, edge_index)
        
        x_pooled = global_mean_pool(x, graph_batch.batch) # (batch_size, out_dim)
        return x_pooled

    def create_grid_edges(self, rows, cols):
        edge_index = []
        for r in range(rows):
            for c in range(cols):
                node = r * cols + c
                if r > 0: edge_index.append([node, (r - 1) * cols + c])
                if c > 0: edge_index.append([node, r * cols + (c - 1)])
                if r < rows - 1: edge_index.append([node, (r + 1) * cols + c])
                if c < cols - 1: edge_index.append([node, r * cols + (c + 1)])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index

# --- 3. Main Model ---

class GOSwinRaw(nn.Module):
    def __init__(self, raw_out_dim=256, swin_out_dim=256, embed_dim=256):
        super().__init__()
        self.raw_encoder = RawWav2VecConformer(out_dim=raw_out_dim)
        self.graph_swin_encoder = GraphSwinEncoder(out_dim=swin_out_dim)
        
        self.fusion_dim = raw_out_dim + swin_out_dim
        
        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5), # <-- Increased from 0.3 to 0.5 for better regularization
            nn.Linear(512, embed_dim)
        )

    def forward(self, raw_waveforms_padded, raw_attention_mask, spec_tensors):
        raw_embedding = self.raw_encoder(raw_waveforms_padded, raw_attention_mask)
        graph_swin_embedding = self.graph_swin_encoder(spec_tensors)
        
        final_embedding = torch.cat([raw_embedding, graph_swin_embedding], dim=1)
        
        embedding = self.head(final_embedding)
        return embedding

