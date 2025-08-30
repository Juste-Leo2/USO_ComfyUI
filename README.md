# USO Nodes for ComfyUI

This project provides custom nodes for ComfyUI to integrate **USO**, a powerful image generation framework based on the FLUX.1 architecture, developed by ByteDance. The goal is to make USO's advanced capabilities for content and style reference accessible within the ComfyUI ecosystem.

This implementation is based on the original research and code from the [ByteDance/USO GitHub repository](https://github.com/bytedance/USO).

## Features

This suite of custom nodes successfully implements the core logic of the USO pipeline:

*   **✅ `USO Model Loader` Node:**
    *   Loads all necessary models: FLUX.1, VAE, DiT LoRA, and Projector from local `.safetensors` files.
    *   Integrates all required text and vision encoders: T5-XXL, CLIP, and SigLIP.
    *   Handles automatic downloading and caching of the encoders from Hugging Face if they are not found locally.
    *   Packages all components into a single `USO_MODEL` output for easy use in workflows.

*   **✅ `USO Sampler` Node:**
    *   Provides a complete text-to-image generation pipeline.
    *   Supports optional image inputs for **content reference** and **style reference**, leveraging the full power of the USO model.
    *   Includes standard generation parameters like seed, steps, dimensions, and guidance.

## Current Status & The Main Challenge

**Project Status: Functionally Complete, but Facing an Optimization Challenge**

The nodes are functionally complete. The entire pipeline, from model loading and text/image encoding to the final sampling loop, has been implemented and thoroughly debugged.

The primary and currently blocking challenge is the **high overall memory consumption**. The complete USO pipeline, which includes multiple large models working in concert, requires a substantial amount of VRAM. On most consumer-grade GPUs, the workflow currently fails with a `torch.OutOfMemoryError` either during model loading or at the very beginning of the sampling process.

Therefore, this project should be considered a **successful proof-of-concept that is currently facing a hardware/optimization barrier.** The code works, but the memory footprint needs to be reduced to make it accessible.

## A Note from the Author

It's important to note that I am not an expert developer. I'm an enthusiast who heavily utilizes Large Language Models (LLMs) to assist in my coding projects. I've pushed this project as far as I can with these tools and my current knowledge. This project is a testament to what's possible with modern AI-assisted development, but it has now reached a point where deeper expertise in model optimization is needed to move forward.

## Call for Contributions

This is where the community comes in!

The logical foundation is solid, but the project needs optimization to be usable by a wider audience. I am opening this project to the open-source community for collaboration.

The main goal is to find creative ways to reduce the memory footprint. If you have experience with model optimization, memory management in PyTorch, or hardware with high VRAM for testing, your contributions would be invaluable.

Please feel free to fork the repository, experiment with solutions, and submit a Pull Request.

## Installation

1.  Clone this repository into your `ComfyUI/custom_nodes/` directory:
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    ```
2.  Restart ComfyUI.

## Acknowledgments

This project would not be possible without the groundbreaking work from ByteDance on the original **USO** model. All credit for the model architecture and its capabilities goes to the original authors.