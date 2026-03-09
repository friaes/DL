import torch.nn
import torch.nn.functional as F
from einops import rearrange

from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator


class Model(BaseModel):
    def __init__(self, vocabulary, embedding_dim, num_layers):
        super().__init__(vocabulary=vocabulary)

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.image_encoder = ImageEncoder(embedding_dim=self.embedding_dim)
        self.caption_generator = CaptionGenerator(vocabulary_size=len(self.vocabulary),
                                                  embedding_dim=self.embedding_dim,
                                                  hidden_dim=self.embedding_dim,
                                                  num_layers=self.num_layers)


class ImageEncoder(BaseImageEncoder):
    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        self.projection = torch.nn.Sequential(
            torch.nn.LayerNorm(self.dinov2.embed_dim),
            torch.nn.Linear(self.dinov2.embed_dim, embedding_dim),
            torch.nn.ReLU()
        )

    def forward(self, image):
        image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        tokens = self.dinov2.get_intermediate_layers(image, n=1, reshape=True, return_class_token=True)[0]
        cls_token = tokens[1]
        return self.projection(cls_token)

    def freeze(self):
        for param in self.dinov2.parameters():
            param.requires_grad = False


class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, num_layers):
        super().__init__(vocabulary_size=vocabulary_size)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.embedding_dim),
            torch.nn.Dropout(0.5)
        )

        self.gru = torch.nn.GRU(input_size=self.embedding_dim,
                                hidden_size=self.hidden_dim,
                                num_layers=self.num_layers,
                                batch_first=True)

        self.to_logits = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.vocabulary_size)

    def freeze(self):
        pass

    def _get_embeddings(self, caption_indices):
        return self.embedding(caption_indices)

    def forward(self, encoded_image, caption_indices, hidden_state=None):
        if hidden_state is None and encoded_image is not None:
            hidden_state = encoded_image.unsqueeze(0).repeat(self.num_layers, 1, 1)

        embeddings = self._get_embeddings(caption_indices)
        
        output, hidden_state = self.gru(embeddings, hx=hidden_state)
        logits = self.to_logits(output)
        logits = rearrange(logits, 'batch sequence_length vocabulary_size -> batch vocabulary_size sequence_length')

        return {'logits': logits, 'indices': logits.argmax(dim=-1), 'hidden_state': hidden_state}

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        caption_indices = [sos_token_index]

        for _ in range(max_length):
            input_tensor = torch.tensor([caption_indices], device=encoded_image.device)
            out = self.forward(encoded_image, input_tensor)

            # Get next token from the last time step of output
            next_token = out['logits'][:, :, -1].argmax(dim=1).item()
            caption_indices.append(next_token)

            if next_token == eos_token_index:
                break

        return caption_indices[1:]  # drop SOS

