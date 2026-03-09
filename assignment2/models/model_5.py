import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator

class Model(BaseModel):
    def __init__(self, vocabulary, embedding_dim=512, num_layers=6, nhead=8):
        super().__init__(vocabulary=vocabulary)
        
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        self.image_encoder = ImageEncoder(embedding_dim=embedding_dim)
        self.caption_generator = CaptionGenerator(vocabulary_size=len(vocabulary),
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
        tokens = self.dinov2.get_intermediate_layers(image, n=1, return_class_token=True)[0]
        return self.projection(tokens[1])

    def freeze(self):
        for param in self.dinov2.parameters():
            param.requires_grad = False

class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, embedding_dim, num_layers, nhead, hidden_dim=None):
        super().__init__(vocabulary_size=vocabulary_size)

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.hidden_dim = hidden_dim if hidden_dim is not None else embedding_dim * 4

        self.embedding = nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.embedding_dim)
        self.dropout = nn.Dropout(0.3)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.nhead,
            dim_feedforward=self.hidden_dim,
            dropout=0.2,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.positional_encoding = nn.Parameter(torch.randn(512, self.embedding_dim) * 0.02)  # max length 512

        self.to_logits = nn.Linear(in_features=self.embedding_dim, out_features=self.vocabulary_size)

    def freeze(self):
        pass

    def _get_embeddings(self, caption_indices):
        emb = self.embedding(caption_indices)
        emb = self.dropout(emb)
        return emb

    def forward(self, encoded_image, caption_indices):
        caption_emb = self._get_embeddings(caption_indices)  # [batch, seq_len, emb_dim]

        # Prepare image token: [batch, 1, emb_dim]
        image_token = encoded_image.unsqueeze(1)

        # Concatenate image token as first token
        tokens = torch.cat([image_token, caption_emb], dim=1)  # [batch, seq_len+1, emb_dim]

        pos_enc = self.positional_encoding[:tokens.size(1), :].unsqueeze(0)  # [1, seq_len+1, emb_dim]
        tokens = tokens + pos_enc

        tokens = tokens.transpose(0, 1) # Transformer expects [seq_len+1, batch, emb_dim]

        # Generate attention mask to prevent attending to future tokens (causal mask)
        seq_len_plus1 = tokens.size(0)
        attn_mask = torch.triu(torch.ones(seq_len_plus1, seq_len_plus1, device=tokens.device), diagonal=1).bool()

        out = self.transformer(tokens, mask=attn_mask)  # [seq_len+1, batch, emb_dim]

        # Discard the image token output, keep only caption outputs
        out = out[1:]  # [seq_len, batch, emb_dim]
        out = out.transpose(0, 1)  # [batch, seq_len, emb_dim]

        logits = self.to_logits(out)  # [batch, seq_len, vocab]
        logits = rearrange(logits, 'batch sequence_length vocabulary_size -> batch vocabulary_size sequence_length')

        return {'logits': logits, 'indices': logits.argmax(dim=-1)}

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        device = encoded_image.device
        caption_indices = [sos_token_index]

        for _ in range(max_length):
            input_tensor = torch.tensor([caption_indices], device=device)  # [1, T]
            out = self.forward(encoded_image, input_tensor)
            
            next_token = out['logits'][:, :, -1].argmax(dim=1).item()
            caption_indices.append(next_token)

            if next_token == eos_token_index:
                break

        return caption_indices[1:]  # drop SOS
