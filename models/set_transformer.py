import torch.nn as nn
from typing import Optional
from models.nets import SAB, ISAB, PMA, FFN

class SetTransformer(nn.Module):
    """
    Set Transformer (Lee et al., 2019) | (https://arxiv.org/abs/1810.00825).

    A neural network architecture designed to process and reason about set-structured data.

    Attrs:
        input_proj (nn.Linear): Initial linear projection layer.
        encoder    (nn.Module): Stacked SAB/ISAB layers.
        decoder    (nn.Module): Stacked SAB layers.
        pma        (nn.Module): PMA layer for pooling.
        task_head  (nn.Module): Feedforward MLP.
    """
    def __init__(
        self,
        input_dim  : int,
        output_dim : int,
        embed_dim  : int = 128,
        num_heads  : int = 4,
        num_sabs   : int = 2,
        num_induce : int = None,
        num_seeds  : int = 1,
        layer_norm : bool = False,
    ):
        """
        Args:
            input_dim  : Feature dimensions of input.
            output_dim : Feature dimensions of output.
            embed_dim  : Embedding dimension throughout the network.
            num_heads  : Number of attention heads.
            num_sabs   : Number of stacked SAB/ISAB layers in encoder and decoder.
            num_induce : Number of inducing points. SAB: None, ISAB: int > 0.
            num_seeds  : Number of seed vectors in PMA layer.
            layer_norm : Whether to apply LayerNorm in MAB(s).
        """
        super().__init__()

        # Input embedding
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Encoder: SAB or ISAB layers
        if num_induce is None:
            self.encoder = nn.Sequential(*[SAB(embed_dim, num_heads, layer_norm) for _ in range(num_sabs)])
        else:
            self.encoder = nn.Sequential(*[ISAB(embed_dim, num_heads, num_induce, layer_norm) for _ in range(num_sabs)])

        # # Decoder: PMA
        # self.decoder = nn.Sequential(
        #     nn.Dropout(),
        #     PMA(embed_dim, num_heads, num_seeds),
        #     nn.Dropout(),
        # )
        
        # Decoder: PMA
        self.decoder = nn.Sequential(
            nn.Dropout(),
            PMA(embed_dim, num_heads, num_seeds),
            SAB(embed_dim,num_heads,layer_norm),
            SAB(embed_dim,num_heads,layer_norm),
            nn.Dropout(),
        )

        # Task Head MLP
        self.task_head = nn.Sequential(
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, x):
        # x : (B, N, 3)
        x = self.input_proj(x)   # (B, N, embed_dim)
        x = self.encoder(x)      # (B, N, embed_dim)
        x = self.decoder(x)      # (B, num_seeds, embed_dim)

        # only if num_seeds == 1
        if x.shape[1] == 1:
            x = x.squeeze(1)     # (B, embed_dim)

        x = self.task_head(x)    # (B, output_dim) or (B, num_seeds, output_dim)
        return x