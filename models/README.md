# LLM Models Directory

This folder is the designated location for Large Language Model weights (such as `.gguf` files).

**Why is this folder empty in the repo?**
Model weights are typically several gigabytes in size. Git is not designed to handle large binary files, so they are ignored via `.gitignore` to keep the repository lightweight and fast to clone.

**Required Action Before Running:**
Before starting N.O.V.A., you must download the necessary model(s) and place them directly in this folder. 

Expected files based on standard configuration:
- `nova_model_lora.gguf`
- `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (or whichever base model you are using)

Ensure your `settings.json` points to the correct filenames located in this directory.