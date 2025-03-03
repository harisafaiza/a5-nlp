# A5 DPO Notebook

This repository contains a Python script for training a **Direct Preference Optimization (DPO) model** using Hugging Face's `transformers` and `trl` libraries. The script loads a dataset, fine-tunes a causal language model, and uploads the trained model to Hugging Face Hub.

## **Features**
- Uses **Hugging Face Hub** authentication for model management.
- Loads the **Anthropic/hh-rlhf** dataset for training.
- Fine-tunes **facebook/opt-1.3b** (can be replaced with a smaller model if needed).
- Implements **Direct Preference Optimization (DPO)** for alignment training.
- Saves the trained model and pushes it to **Hugging Face Hub**.

## **Installation**
Make sure you have Python installed, then install the required dependencies:

```bash
pip install torch transformers datasets trl huggingface_hub
```

## **Usage**
### **1. Set Up Hugging Face Authentication**
Before running the script, log in to Hugging Face Hub using:
```bash
huggingface-cli login
```
Or set your token in the script:
```python
export HF_TOKEN="your_huggingface_token"
```

### **2. Run the Script**
```bash
python A5_DPO.py
```

## **Customization**
- Change `model_name` to a different model if needed:
  ```python
  model_name = "gpt2"
  ```
- Adjust training parameters in `DPOConfig`:
  ```python
  dpo_config = DPOConfig(
      beta=0.1,
      per_device_train_batch_size=4,
      save_steps=500,
      output_dir="./my_results",
  )
  ```

## **Results & Model Upload**
- The trained model is saved in `./results`.
- It is automatically pushed to **Hugging Face Hub** under `your_huggingface_repo_name`.
- Visit [huggingface.co](https://huggingface.co/) to view the model.

## **Troubleshooting**
- **Authentication Errors**: Ensure your Hugging Face token is valid.
- **CUDA Out of Memory**: Reduce `per_device_train_batch_size`.
- **Dataset Not Found**: Ensure `Anthropic/hh-rlhf` is accessible.


