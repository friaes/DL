# Deep Learning (2025) — Coursework Repository

Tutorials and graded assignments for a Master's-level Deep Learning course, implemented in PyTorch.
The repository spans the full arc of the course: from NumPy/PyTorch fundamentals through CNNs, RNNs and
attention, up to Vision Transformers, self-supervised learning, VAEs and diffusion models.

The two graded assignments are the substantial pieces of work:

| Assignment | Task | Highlight |
|---|---|---|
| **Assignment 1** | Pneumonia detection from chest X-rays (binary classification) | MLP vs. CNN + regularization study |
| **Assignment 2** | Image captioning on Flickr8k | Six-model progression, RNN → cross-attention over DINOv2 spatial tokens |

---

## Repository structure

```
.
├── assignment1/                 # Chest X-ray classification (MLP vs CNN)
│   ├── Assignment1.ipynb
│   ├── Assignment1.html         # rendered submission
│   ├── DL_assignment.pdf        # task description
│   └── data.zip                 # dataset
│
├── assignment2/                 # Image captioning on Flickr8k
│   ├── DL_2025_Assignment_2.pdf # task description
│   ├── report.ipynb / .html     # graded report (curves, BLEU, analysis)
│   ├── report_utils/            # loss curves (CSV) + 5 test images
│   ├── colab_run.ipynb          # Colab driver (mount Drive, install, train)
│   ├── models/ , configs/       # working copies of the edited files
│   └── assignment_2/            # the full training framework (see below)
│
└── Tutorial 01 … Tutorial 13/   # weekly lab notebooks (template + solution)
```

---

## Assignment 2 — Image Captioning on Flickr8k

The core of the repository. A caption generator is built up over six models, each one changing a single
design decision so the effect is measurable in isolation. All models are trained with teacher forcing and
cross-entropy loss (padding ignored), and evaluated with corpus BLEU-1…4 on the validation split.

### Model progression

| Model | Image encoder | Decoder | Image conditioning |
|---|---|---|---|
| 1 | ResNet-18, **trained from scratch** | RNN (2 layers) | image code as first input token |
| 2 | **DINOv2 ViT-B/14, frozen** (`<CLS>`) | RNN (2 layers) | image code as first input token |
| 3 | DINOv2 frozen (`<CLS>`) | **GRU** (1 layer) | image code as first input token |
| 4 | DINOv2 frozen (`<CLS>`) | GRU (1 layer) | **initializes the decoder hidden state** |
| 5 | DINOv2 frozen (`<CLS>`) | **Transformer** (4 layers, causal mask) | `<CLS>` prepended as a context token |
| 6 | DINOv2 frozen (**256 spatial tokens**) | Transformer decoder (4 layers) | **cross-attention over the 14×14 patch grid** |

Models 2–6 keep the DINOv2 backbone frozen (≈86.6 M frozen parameters); only the projection and decoder
are trained. Images are resized to 256×256 and interpolated to 224×224 inside the models, giving a
16×16 = 256-token patch grid at ViT-B/14.

### Results (validation BLEU)

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|
| Model 1 | 0.4215 | 0.2137 | 0.1112 | 0.0645 |
| Model 2 | 0.5345 | 0.3100 | 0.1610 | 0.0890 |
| Model 3 | 0.5787 | 0.3925 | 0.2540 | 0.1662 |
| Model 4 | 0.6004 | 0.4236 | 0.2813 | 0.1861 |
| Model 5 | 0.6230 | 0.4498 | 0.3119 | 0.2119 |
| **Model 6** | **0.6711** | **0.5094** | **0.3756** | **0.2720** |

The two largest jumps come from replacing the from-scratch CNN with frozen DINOv2 features (1 → 2) and
from letting the decoder attend to *spatial* image tokens instead of a single global vector (5 → 6).

### Framework layout (`assignment2/assignment_2/`)

```
train.py                 # training entry point
evaluate.py              # BLEU evaluation from a checkpoint
metric.py                # corpus BLEU-1..4
models/
  base.py                # BaseModel / BaseCaptionGenerator contracts
  model_1.py … model_6.py
  utils.py               # name → class registry
data/
  dataset.py             # FlickrDataset (image, caption_indices)
  dataloader.py          # collate + padding
  vocabulary.py          # spaCy tokenizer, <PAD>/<SOS>/<EOS>/<UNK>
  transforms.py          # resize 256 + ImageNet normalization
training/
  trainer.py             # train/val loop
  checkpointer.py        # save/load, resume
  logger.py              # Weights & Biases
parsing/                 # CLI args + YAML config parsing
configs/config_model_*.yaml
```

Each model is described entirely by a YAML config (architecture parameters, learning rate, batch size,
data paths). All six use `lr = 1e-4` and `batch_size = 256`; the architectures differ:

| | model_1 | model_2 | model_3 | model_4 | model_5 | model_6 |
|---|---|---|---|---|---|---|
| `embedding_dim` | 128 | 128 | 512 | 256 | 512 | 512 |
| `num_layers` | 2 | 2 | 1 | 1 | 4 | 4 |

### Setup

```bash
conda env create -f environment.yaml     # or environment_cpu.yaml for CPU
conda activate dl_a2
python -m spacy download en_core_web_sm  # required by the vocabulary builder
```

The Flickr8k data is **not** included. Place it under `assignment_2/flickr8k/` as expected by the configs:

```
flickr8k/
├── train_images/            ├── train_captions.txt
├── val_images/              ├── val_captions.txt
                             └── vocabulary_captions.txt
```

### Running

```bash
# train
python train.py --device-id=0 \
                --config-file-path=./configs/config_model_6.yaml \
                --experiment-name=model_6 \
                --num-epochs=10

# resume an interrupted run (reuses the config copied into the checkpoint dir)
python train.py --device-id=0 --experiment-name=model_6 --num-epochs=10 --resume

# evaluate — writes bleu_scores.txt next to the checkpoint
python evaluate.py -d 0 --checkpoint-path=./checkpoints/model_6/model.pth.tar
```

Useful flags: `--no-log` disables Weights & Biases, `--num-workers` sets data-loading workers,
`--seed` fixes reproducibility. Checkpoints land in `checkpoints/<experiment-name>/` and the active
config is copied there automatically so runs stay self-documenting.

Training was run on the University of Bern HPC cluster (SLURM, RTX 3090); `colab_run.ipynb` provides a
Google Colab alternative.

---

## Assignment 1 — Chest X-ray Classification

Binary classification of chest X-rays (*normal* vs. *pneumonia*).

**Task 1** — build the data pipeline, then implement and train a fully connected MLP and a CNN.
**Task 2** — add regularization to the CNN and compare.

| Model | Validation accuracy |
|---|---|
| MLP | 76.76 % |
| CNN (baseline) | 73.24 % |
| CNN + early stopping | 75.32 % |
| CNN + L2 / weight decay | **78.85 %** |

Weight decay gave the best generalization, with early stopping also recovering accuracy over the
unregularized CNN — consistent with the baseline CNN overfitting a fairly small dataset.

---

## Tutorials

Weekly labs, most with both a template and a worked solution notebook.

| # | Topic |
|---|---|
| 01 | Python refresher; dogs-vs-cats with nearest-neighbour classification |
| 02 | PyTorch basics: tensors, `Dataset`/`DataLoader`, linear regression, MNIST |
| 03 | Neural networks: MLPs, loss functions, the training loop |
| 04 | "MNIST calculator" — data prep, inspection, MLP training |
| 05 | Overfitting: regularization and data augmentation |
| 06 | Batch size effects, batch normalization, LR scheduling |
| 07 | Convolutional neural networks; convolution/pooling from first principles; CNN vs. MLP |
| 08 | Character-level text prediction with LSTMs (Shakespeare corpus) |
| 09 | Sequence-to-sequence translation with attention (eng–fra) |
| 10 | Vision Transformers on CIFAR-10; transfer learning; DINOv2 features and segmentation |
| 11 | Self-supervised learning with SimCLR (contrastive learning) |
| 12 | Autoencoders and Variational Autoencoders |
| 13 | Denoising Diffusion Probabilistic Models (DDPM) with a U-Net |

Tutorial 10 is the direct precursor to Assignment 2 — it introduces the frozen DINOv2 features that
models 2–6 are built on.

---

## Tech stack

PyTorch · torchvision · DINOv2 (ViT-B/14) · spaCy · NLTK (BLEU) · Weights & Biases · pandas · matplotlib · Conda · SLURM
