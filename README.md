# Multimodal-Fact-Checking

## Download Embeddings
You need to download embeddings from [here](https://drive.google.com/drive/folders/1DtiAZfqZYm5hsHj9V0VDe2C6acxrkNM7?usp=sharing)

Extract the embeddings directory under "mocheg" directory (i.e. mocheg/embeddings/..)

The directory structure should be as follows:

mocheg/embeddings/model_name/(claim/evidence/multimodal/image)/embedding.npy

More specifically, 
claim + evidence embeddings were saved under "evidence" directory, 
claim embeddings were saved under "claim" directory,
image embeddings were saved under "image" directory,
multimodal embeddings were saved under "multimodal" directory. 

The script "vlm.py" contains all the necessary methods for embedding extraction, as well as inference methods for both LLM and VLM.

## Install Packages
```
pip install -r requirements.txt
```


## Training Probing Classifiers

You need to provide your wandb API key once as follows:
```
wandb.login(key=WANDB_API_KEY)
```

Depending on the number of input embeddings, you should execute either "train_mmfusion.py", "train_fusion.py", or "train_fusion4.py". More specifically, if you have only one input embedding (e.g., a VLM embedding for the claim), execute "train_mmfusion.py". If you have two input embeddings (such as VLM evidence embeddings or an extrinsic fusion of claim and image embeddings), execute "train_fusion.py". For more than two input embeddings, execute "train_fusion4.py".

```
python train_mmfusion.py \
--model_id "google/paligemma-3b" \
--dataset "mocheg" # or "factify2"
--train_embedding_file "train_claim_embeddings.npy" \
--val_embedding_file "val_claim_embeddings.npy" \
--test_embedding_file "test_claim_embeddings.npy" \
--wandb_proj_name "intrinsic_fusion_sweep_v1" \ 
```

or 

```
python train_fusion.py \
--model_id "Qwen/Qwen-VL-Chat-Int4" \
--dataset "mocheg" # or "factify2"
--isclaim True # or False
--train_image_file "image/train_image_embeddings.npy" \
--val_image_file "image/val_image_embeddings.npy" \
--test_image_file "image/test_image_embeddings.npy" \
--train_text_file "claim/train_text_embeddings.npy" \
--val_text_file "claim/val_text_embeddings.npy" \
--test_text_file "claim/test_text_embeddings.npy" \
--wandb_proj_name "extrinsic_fusion_sweep_v1" \ 
```

or 

```
python train_fusion4.py \
--model_id "HuggingFaceM4/idefics2-8b" \
--dataset "mocheg" # or "factify2"
--train_claim_image_file "image/train_image_embeddings.npy" \
--val_claim_image_file "image/val_image_embeddings.npy" \
--test_claim_image_file "image/test_image_embeddings.npy" \
--train_claim_file "claim/train_text_embeddings.npy" \
--val_claim_file "claim/val_text_embeddings.npy" \
--test_claim_file "claim/test_text_embeddings.npy" \
--train_text_image_file "image/train_image_embeddings.npy" \
--val_text_image_file "image/val_image_embeddings.npy" \
--test_text_image_file "image/test_image_embeddings.npy" \
--train_text_file "text/train_text_embeddings.npy" \
--val_text_file "text/val_text_embeddings.npy" \
--test_text_file "text/test_text_embeddings.npy" \
--wandb_proj_name "extrinsic_fusion_sweep_v2" \ 
```

## Fine-tuning a VLM

I followed [this tutorial](https://huggingface.co/blog/paligemma) to fine-tune Paligemma-3b model on Mocheg and Factify2 datasets.
```
python finetune_vlm.py \
--dataset "mocheg" # or "factify2"
--batch 2 \
--lr 2e-5 \
--cache_dir None \
```


## Training Baseline Classifiers 

We have implemented SVM and KNN classifiers that takes the extracted embeddings as input for multi-class classification.

```
python baseline_classifier.py \
--model_id "HuggingFaceM4/idefics2-8b" \
--dataset "mocheg" # or "factify2" \
--classifier "knn" # or "svm" \
--useEvidence False
--train_embedding_file "train_claim_embeddings.npy" \
--test_embedding_file "test_claim_embeddings.npy" \
--train_evidence_file "train_text_embeddings.npy" \
--test_evidence_file "test_text_embeddings.npy" \
```