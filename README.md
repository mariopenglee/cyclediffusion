[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mariopenglee/cyclediffusion)
# [Read this DeepWiki for implementation details](https://deepwiki.com/mariopenglee/cyclediffusion/1-overview)
```mermaid
graph TB
    subgraph "CycleDiffusion System"
        subgraph "Device 0 (cuda:0)"
            subgraph "Diffuser Component"
                TOKENIZER["CLIPTokenizer"]
                TEXT_ENC["CLIPTextModel"]
                UNET["UNet2DConditionModel"]
                VAE["AutoencoderKL"]
                SCHEDULER["DDPMScheduler"]
            end
        end
        
        subgraph "Device 1 (cuda:1)"
            subgraph "Captioner Component"
                P2S["Pix2StructForConditionalGeneration"]
                PROCESSOR["AutoProcessor"]
                FLATTEN["flatten_patches()"]
            end
        end
        
        subgraph "Main Orchestrator"
            CDM["CycleDiffusionModel"]
        end
    end
    
    subgraph "Training Pipeline"
        TRAIN_LOOP["train_cyclediff()"]
        SCALER["GradScaler"]
        WRITER["TensorBoard Writer"]
        OPTIMIZER["Adam Optimizer"]
    end
    
    subgraph "Data Flow"
        INPUT_TEXT["Input Captions"]
        INPUT_IMAGES["Input Images"]
        GEN_IMAGES["Generated Images"]
        RECON_TEXT["Reconstructed Captions"]
        PIXEL_LOSS["Pixel Loss"]
        CAPTION_LOSS["Caption Loss"]
        TOTAL_LOSS["Total Loss"]
    end
    
    %% Main flow
    INPUT_TEXT --> TOKENIZER
    TOKENIZER --> TEXT_ENC
    TEXT_ENC --> UNET
    INPUT_IMAGES --> VAE
    VAE --> UNET
    SCHEDULER --> UNET
    UNET --> VAE
    VAE --> GEN_IMAGES
    
    %% Cycle back
    GEN_IMAGES --> FLATTEN
    FLATTEN --> P2S
    P2S --> RECON_TEXT
    
    %% Loss computation
    UNET --> PIXEL_LOSS
    P2S --> CAPTION_LOSS
    PIXEL_LOSS --> TOTAL_LOSS
    CAPTION_LOSS --> TOTAL_LOSS
    
    %% Training integration
    CDM --> TRAIN_LOOP
    TOTAL_LOSS --> SCALER
    SCALER --> OPTIMIZER
    TRAIN_LOOP --> WRITER
```
## Overview

CycleDiffusion is a deep learning architecture that implements a cycle-consistent training approach for bidirectional text-image generation. The project combines state-of-the-art Stable Diffusion models with Pix2Struct vision-language models to create a self-supervised learning framework that enforces consistency between text and image domains.

## Technical Requirements

The system requires:
- **Multi-GPU setup** (minimum 2 GPUs recommended)
- **CUDA-compatible hardware** with sufficient VRAM
- **PyTorch ecosystem** with diffusers, transformers, and accelerate libraries

## Getting Started

```bash
# Set up environment
export PYTHONPATH="/path/to/cyclediffusion:$PYTHONPATH"

# Launch training
python src/training/train_cyclediffusion.py

# Monitor with TensorBoard
tensorboard --logdir=./runs/
```

## Architecture Highlights

The system demonstrates several advanced ML engineering practices:
- **Modular component design** with clear separation of concerns
- **Device-aware tensor management** for multi-GPU efficiency
- **Comprehensive monitoring** and experiment tracking
- **Robust checkpoint management** with resume capabilities

Checkout the Deep Wiki for more details on implementation:
- [Overview](https://deepwiki.com/mariopenglee/cyclediffusion/1-overview)
- [Core Architecture](https://deepwiki.com/mariopenglee/cyclediffusion/2-core-architecture)
- [Training Systems](https://deepwiki.com/mariopenglee/cyclediffusion/3-training-systems)
- [Data Preparation](https://deepwiki.com/mariopenglee/cyclediffusion/4-data-preparation)
