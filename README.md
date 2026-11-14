# About This Project

This repository is a minimal, from-scratch implementation of the Transformer architecture built with PyTorch, born out of my journey to deeply understand how these models work under the hood.

Instead of relying on high-level wrappers, this project focuses on clean, educational code using native PyTorch tensor operations to demystify the core components of the Transformer, such as Multi-Head Attention, Positional Encoding, and Feed-Forward Networks.

# Key Features:

- Built with PyTorch: Leverages PyTorch's dynamic computation graphs and native torch.nn modules, making the code intuitive and highly educational.

- Two Tokenization Variants: Includes two separate versions to explore different text processing strategies:

  - BPE Version: Utilizes Byte Pair Encoding (BPE) for subword tokenization, similar to modern LLMs like GPT.

  - Byte-level Version: Operates directly on raw bytes, exploring a more granular encoding approach.

- Minimal & Educational: Stripped of complex engineering boilerplate, the code focuses purely on the model architecture and data flow, making it an ideal reference for learning both Transformers and PyTorch.
