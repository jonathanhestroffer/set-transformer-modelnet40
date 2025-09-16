import math
import torch 
import torch.nn as nn

class FFN(nn.Module):
    """
    FeedForward Network (FFN).
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim,embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim,embed_dim),
        )
    def forward(self, x):
        return self.ffn(x)

class MAB(nn.Module):
    """
    Multihead Attention Block (MAB).

    An adaptation of the encoder block from the Transformer architecture (Vaswani et al., 2017).
    
    Omits: 
        - positional encodings for permutation invariance.
        - dropout layers for simplicity.

    Attrs:
        embed_dim      (int): Dimensionality of input embeddings.
        num_heads      (int): Number of attention heads.
        qkv_proj (nn.Linear): Linear layer to project input to Q, K, V.
        out_proj (nn.Linear): Linear layer to project concatenated heads.
        rFF      (nn.Module): Row-wise feedforward network.
        norm1 (nn.LayerNorm): Layer normalization after attention.
        norm2 (nn.LayerNorm): Layer normalization after feedforward network.
    """
    def __init__(
        self, 
        embed_dim  : int,
        num_heads  : int,
        layer_norm : bool = False
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        # linear projection for Q, K, V for all heads
        self.q_proj   = nn.Linear(embed_dim, embed_dim)
        self.kv_proj  = nn.Linear(embed_dim, 2*embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # row-wise feedforward layer
        self.rFF = FFN(embed_dim)

        # layer norms
        self.ln0 = nn.LayerNorm(embed_dim) if layer_norm else nn.Identity()
        self.ln1 = nn.LayerNorm(embed_dim) if layer_norm else nn.Identity()

    def forward(self, x, y):
        """
        H   = LayerNorm(X + MultiHeadAttention(X, Y, Y))
        MAB = LayerNorm(H + rFeedForward(H))

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, embed_dim).
        """
        batch_size, x_length, embed_dim = x.size()
        batch_size, y_length, embed_dim = y.size()

        # Linear projections
        q    = self.q_proj(x)
        kv   = self.kv_proj(y)
        k, v = kv.chunk(2, dim=-1)

        # Get correct shapes for attention
        # shapes: (batch_size, num_heads, seq_length, head_dim)
        q = q.view(batch_size, x_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, y_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, y_length, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn_logits  = torch.matmul(q, k.transpose(-2, -1))
        attn_logits  = attn_logits / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        outputs      = torch.matmul(attn_weights, v)

        # Concatenate heads
        output = outputs.transpose(1, 2).contiguous().view(batch_size, x_length, embed_dim)
        
        # Final linear projection
        output = self.out_proj(output)

        # Add & Norm + Feedforward
        h = self.ln0(x + output)      # Add Residual & Norm
        return self.ln1(h + self.rFF(h)) # Feedforward, Add Residual & Norm

class SAB(nn.Module):
    """
    Set Attention Block (SAB).

    SAB(X) = MAB(X, X)
    """
    def __init__(self, embed_dim: int, num_heads: int, layer_norm: bool = False):
        super().__init__()
        self.mab = MAB(embed_dim, num_heads, layer_norm)
 
    def forward(self, x):
        return self.mab(x, x)
     
class ISAB(nn.Module):
    """
    Induced Set Attention Block (ISAB).

    ISAB(X) = MAB(X, H)
       H    = MAB(I, X)
    
    Introduces learnable inducing points to reduce complexity from O(n^2) to O(nm),
    where n = input set size and m = number of inducing points.
    """
    def __init__(self, embed_dim: int, num_heads: int, m: int, layer_norm: bool = False):
        super().__init__()
        self.mab_ind = MAB(embed_dim, num_heads, layer_norm)
        self.mab_out = MAB(embed_dim, num_heads, layer_norm)
        self.i = nn.Parameter(torch.Tensor(1, m, embed_dim))
        nn.init.xavier_uniform_(self.i)
 
    def forward(self, x):
        batch_size = x.shape[0]
        i = self.i.repeat(batch_size, 1, 1) # expand to batch
        h = self.mab_ind(i, x)
        return self.mab_out(x, h)

class PMA(nn.Module):
    """
    Pooling by Multihead Attention (PMA).

    PMA(X) = MAB(S, X), where S are learnable seed vectors

    This module aggregates a set of input embeddings into a fixed-size output set using attention mechanisms.
    """
    def __init__(self, embed_dim: int, num_heads: int, k: int = 1, layer_norm: bool = False):
        super().__init__()
        self.mab = MAB(embed_dim, num_heads, layer_norm)
        self.s = nn.Parameter(torch.Tensor(1, k, embed_dim))
        nn.init.xavier_uniform_(self.s)

    def forward(self, x):
        batch_size = x.shape[0]
        s = self.s.repeat(batch_size, 1, 1) # expand to batch
        return self.mab(s, x)