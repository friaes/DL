import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator


class Model(BaseModel):
    def __init__(self, vocabulary, embedding_dim=512, num_layers=4, nhead=8):
        super().__init__(vocabulary=vocabulary)
        self.embedding_dim = embedding_dim
        self.image_encoder = ImageEncoder(embedding_dim)
        self.caption_generator = CaptionGenerator(
            vocabulary_size=len(vocabulary),
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            nhead=nhead
        )


class ImageEncoder(BaseImageEncoder):
    def __init__(self, embedding_dim):
        super().__init__()
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.projection = nn.Sequential(
            nn.LayerNorm(self.dinov2.embed_dim),
            nn.Linear(self.dinov2.embed_dim, embedding_dim),
            nn.ReLU()
        )
        self.freeze()

    def forward(self, image):
        image = F.interpolate(image, size=(224, 224), mode="bilinear")
        tokens = self.dinov2.get_intermediate_layers(image, n=1, return_class_token=True)[0]  # [B, T, D]
        return self.projection(tokens[0])  # spatial tokens

    def freeze(self):
        for p in self.dinov2.parameters():
            p.requires_grad = False


class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, embedding_dim, num_layers, nhead, hidden_dim=None):
        super().__init__(vocabulary_size=vocabulary_size)
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.hidden_dim = hidden_dim or embedding_dim * 4

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.dropout = nn.Dropout(0.3)

        decoder_layer = TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=self.hidden_dim,
            dropout=0.2,
            batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.positional_encoding = nn.Parameter(torch.randn(512, embedding_dim) * 0.02)
        self.to_logits = nn.Linear(embedding_dim, vocabulary_size)

    def freeze(self):
        pass

    def _get_embeddings(self, caption_indices):
        emb = self.embedding(caption_indices)
        return self.dropout(emb)

    def forward(self, encoded_image, caption_indices):
        tgt_emb = self._get_embeddings(caption_indices)  # [B, T, D]

        # Add positional encoding
        pe = self.positional_encoding[:tgt_emb.size(1), :].unsqueeze(0)
        tgt = tgt_emb + pe

        # Causal mask for autoregression
        T = tgt.size(1)
        tgt_mask = torch.triu(torch.ones((T, T), device=tgt.device), diagonal=1).bool()

        output = self.decoder(
            tgt=tgt,
            memory=encoded_image,  # spatial features
            tgt_mask=tgt_mask
        )  # [B, T, D]

        logits = self.to_logits(output)  # [B, T, V]
        logits = rearrange(logits, 'batch sequence_length vocabulary_size -> batch vocabulary_size sequence_length')
        return {'logits': logits, 'indices': logits.argmax(dim=-1)}

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        device = encoded_image.device
        caption_indices = [sos_token_index]

        for _ in range(max_length):
            input_tensor = torch.tensor([caption_indices], device=device)  # shape: [1, T]
            out = self.forward(encoded_image, input_tensor)

            next_token = out['logits'][:, :, -1].argmax(dim=1).item()
            caption_indices.append(next_token)

            if next_token == eos_token_index:
                break

        return caption_indices[1:]  # remove SOS

