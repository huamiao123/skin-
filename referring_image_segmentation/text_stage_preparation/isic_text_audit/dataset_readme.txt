# SkinDualGen

**Prompt-driven diffusion for jointly generating dermoscopic images and lesion masks.**

SkinDualGen fine-tunes Stable Diffusion 2 with LoRA to synthesize paired skin-lesion images and segmentation masks from class-aware prompts. The generated data can then be evaluated in real, synthetic, and hybrid training regimes for classification and segmentation.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-deep%20learning-EE4C2C?logo=pytorch&logoColor=white)
![Diffusers](https://img.shields.io/badge/Hugging%20Face-Diffusers-FFD21E)

## Overview

![SkinDualGen pipeline](skin.png)

The project covers the complete experimental loop:

1. prepare dermoscopic images, masks, labels, and GPT-generated descriptions;
2. fine-tune Stable Diffusion 2 with LoRA;
3. generate class-conditioned image–mask pairs;
4. assemble real, synthetic, and hybrid datasets;
5. evaluate downstream classifiers and segmentation models;
6. test cross-dataset robustness on HAM10000;
7. inspect classifier behavior with Grad-CAM, saliency, and occlusion maps.

## Supported experiments

| Task | Models / modes |
|---|---|
| Classification | DenseNet-121, ResNet-34, SqueezeNet 1.1, Swin Transformer |
| Segmentation | U-Net, DeepLabV3+ |
| Training data | real, synthetic, hybrid |
| Robustness | external evaluation on HAM10000 |
| Explainability | Grad-CAM, saliency, occlusion |

The seven lesion categories used by the dataset preparation scripts are basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, squamous cell carcinoma, and vascular lesion.

## Results preview

<table>
  <tr>
    <td><img src="comparison1.png" alt="Generated image and mask comparison 1"></td>
    <td><img src="comparison2.png" alt="Generated image and mask comparison 2"></td>
  </tr>
</table>

![Classification and segmentation results](table2.png)

![HAM10000 robustness results](table3.png)

![Classification explainability maps](vis.png)

## Environment

The original experiments use Hugging Face Diffusers, Accelerate, PyTorch, and a CUDA-capable GPU.

```bash
git clone https://github.com/JaspinXu/SkinDualGen.git
cd SkinDualGen
python -m pip install -e ./diffusers
python -m pip install -r ./diffusers/examples/text_to_image/requirements.txt
accelerate config
```

Authenticate with Hugging Face only if the selected base model or workflow requires it:

```bash
huggingface-cli login
```

## Dataset

```bash
curl -L -o isic_gpt.zip \
  https://www.kaggle.com/api/v1/datasets/download/zhaobinxu/skin-cancer-isic-and-gpt-based-descriptions
unzip isic_gpt.zip -d dataset/
```

Update all example paths below for your local machine.

## Train the joint image–mask generator

```bash
accelerate launch --mixed_precision="bf16" train_text_to_image_lora_ft_2_m.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2" \
  --local_dataset_path="/path/to/dataset/isic_gpt" \
  --resolution=256 --train_batch_size=4 --num_train_epochs=100 \
  --checkpointing_steps=10000 --learning_rate=1e-4 \
  --lr_scheduler="constant" --lr_warmup_steps=0 --seed=42 \
  --output_dir="/path/to/checkpoints" \
  --validation_prompt="a dermoscopic lesion photo of dermatofibroma for skin cancer diagnosis" \
  --rank=4 --use_8bit_adam
```

## Generate an expanded dataset

```bash
python Inference_percent_skin_2_m.py \
  --lora_path="/path/to/checkpoints/checkpoint-10000" \
  --modes="basal cell carcinoma,dermatofibroma,melanoma,nevus,pigmented benign keratosis,squamous cell carcinoma,vascular lesion" \
  --percent_list=1.0 --resolution=512 \
  --existing_images_base_directory="/path/to/dataset/isic_gpt/train" \
  --save_dir="/path/to/dataset/generated" \
  --prompt_structure="a dermoscopic lesion photo of {class_name} for skin cancer diagnosis" \
  --copy_flag=0
```

## Downstream evaluation

```bash
python evaluate_new_classification.py \
  --data_path="/path/to/train" --test_path="/path/to/test" \
  --output_path="/path/to/results/classification"

python evaluate_new_segmentation.py \
  --data_path="/path/to/train" --test_path="/path/to/test" \
  --train_mask_path="/path/to/train_masks" --test_mask_path="/path/to/test_masks" \
  --output_path="/path/to/results/segmentation" --loss_type=dice
```

Use each evaluator's CLI options to select the model and test-only modes. The scripts are designed to compare equivalent real, synthetic, and hybrid dataset layouts.

## Repository guide

```text
train_text_to_image_lora_ft_2_m.py  Joint image–mask LoRA fine-tuning
Inference_percent_skin_2_m.py       Paired dataset generation
evaluate_new_classification.py      Classification experiments
evaluate_new_segmentation.py        Segmentation experiments
vis/                                Explainability tooling
dataset/                            Dataset helpers and local data layout
expe/                               Experiment outputs
diffusers/                          Experiment-time Diffusers source snapshot
```

## Responsible-use note

This is a research codebase for data augmentation experiments. Generated images and model outputs are not medical diagnoses and must not be used for clinical decisions without appropriate validation, governance, and expert review.

