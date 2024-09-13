from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig, AutoProcessor, AutoModelForVision2Seq
from transformers import TrainingArguments, Trainer
import torch
from peft import get_peft_model, LoraConfig, PeftModel
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from PIL import Image
from tqdm import tqdm
import wandb
import os
import argparse
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split

system_prompt = "Assess the factuality of the following claim by considering image and evidence. You must only answer \"supported\", \"refuted\" or \"not enough info\"."
label2id = {"NEI":0, "not enough info":0, "supported":2, "refuted":1, 
            "Support_Multimodal":2, "Support_Text":2, "Insufficient_Text":0, "Insufficient_Multimodal":0, "Refute":1}
id2label = {0:"NEI", 0:"not enough info", 2:"supported", 1:"refuted"}


parser = argparse.ArgumentParser() 
parser.add_argument('--dataset', default='factify2', type=str, help='dataset name')
parser.add_argument('--batch', default=2, type=int, help='batch size')
parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
parser.add_argument('--cache_dir', default=None, type=str, help='hf model cache dir')
args = parser.parse_args()

SKIP_TRAIN = False
OUTPUT_DIR = f"paligemma_{args.dataset}_2"
isTestset = False

def filter_text(text, context_len=2500):
    words = text.split()  
    if len(words) > context_len:
        first_k_words = words[:context_len]  
        truncated_text = ' '.join(first_k_words) 
        return truncated_text
    return text

def get_image_url(row):
    if args.dataset == "mocheg":
        images = row["img_evidences"].split(";")
        return images[0]
    else:
        return str(row["index"]) + "_claim.jpg"

def generate_training_prompt(claim: str, evidence: str, label: str, system_prompt: str = system_prompt):
    if isTestset:
        label = ""
        evidence = filter_text(evidence, context_len=2048) # 3072
        prompt =  f"""{system_prompt}
Claim: {claim.strip()}
Evidence: {evidence.strip()}
""".strip()
    else:
        evidence = filter_text(evidence, context_len=512) # 768
        prompt =  f"""{system_prompt}
Claim: {claim.strip()}
Evidence: {evidence.strip()}
""".strip() 
    
    return prompt

def generate_text(data_point, img_path):
    text = generate_training_prompt(claim=data_point["Claim"], evidence=data_point['Evidence'], label=data_point['cleaned_truthfulness'], system_prompt=system_prompt)
    return {
        "image" : img_path + get_image_url(data_point),#Image.open(img_path + get_image_url(data_point)).convert("RGB"),
        #"image" : get_image_url(data_point),
        "label" : data_point["cleaned_truthfulness"],
        "prompt" : text
    }

def process_dataset(data: Dataset, img_path:str):
    return (
        data.shuffle(seed=42)
        .map(lambda x: generate_text(x, img_path))
        .remove_columns(["Claim", "Evidence",  "cleaned_truthfulness",  "img_evidences", "index"]) #"__index_level_0__",
    )


if args.dataset == "mocheg":
    train_df = pd.read_csv(f"{args.dataset}/train.csv", encoding="utf-8")
    train_df = train_df[["Claim", "Evidence", "img_evidences", "cleaned_truthfulness"]]
    train_df.dropna(inplace=True)

    val_df = pd.read_csv(f"{args.dataset}/val.csv", encoding="utf-8")
    val_df = val_df[["Claim", "Evidence", "img_evidences", "cleaned_truthfulness"]]
    val_df.dropna(inplace=True)

    test_df = pd.read_csv(f"{args.dataset}/test.csv", encoding="utf-8")
    test_df = test_df[["Claim", "Evidence", "img_evidences", "cleaned_truthfulness"]]
    test_df.dropna(inplace=True)

    dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df),
    "test": Dataset.from_pandas(test_df)
    })

    dataset["train"] = process_dataset(dataset["train"], img_path="../multimodal-fc/mocheg/train/images/")
    dataset["validation"] = process_dataset(dataset["validation"], img_path="../multimodal-fc/mocheg/val/images/")
    isTestset = True
    dataset["test"] = process_dataset(dataset["test"], img_path="../multimodal-fc/mocheg/test/images/")

elif args.dataset == "factify2":
    train_df = pd.read_csv(f"{args.dataset}/train_baseline.csv", encoding="utf-8", sep="\t")
    train_df = train_df[["index", "Claim", "claim_image", "Evidence",  "cleaned_truthfulness"]]
    train_df['cleaned_truthfulness'] = train_df['cleaned_truthfulness'].replace('NEI', 'not enough info')
    train_df = train_df.rename(columns={'claim_image': 'img_evidences'})
    train_df.dropna(inplace=True)

    val_df = pd.read_csv(f"{args.dataset}/val_baseline.csv", encoding="utf-8", sep="\t")
    val_df = val_df[["index", "Claim", "claim_image", "Evidence",  "cleaned_truthfulness"]]
    val_df['cleaned_truthfulness'] = val_df['cleaned_truthfulness'].replace('NEI', 'not enough info')
    val_df = val_df.rename(columns={'claim_image': 'img_evidences'})
    val_df.dropna(inplace=True)

    test_df = pd.read_csv(f"{args.dataset}/test_baseline.csv", encoding="utf-8", sep="\t")
    test_df = test_df[["index", "Claim", "claim_image", "Evidence",  "cleaned_truthfulness"]]
    test_df['cleaned_truthfulness'] = test_df['cleaned_truthfulness'].replace('NEI', 'not enough info')
    test_df = test_df.rename(columns={'claim_image': 'img_evidences'})
    test_df.dropna(inplace=True)

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })

    dataset["train"] = process_dataset(dataset["train"], img_path=f"{args.dataset}/images/train/")
    dataset["validation"] = process_dataset(dataset["validation"], img_path=f"{args.dataset}/images/train/")
    isTestset = True
    dataset["test"] = process_dataset(dataset["test"], img_path=f"{args.dataset}/images/val/")




model_id = "google/paligemma-3b-pt-224"
cache_dir = args.cache_dir
processor = PaliGemmaProcessor.from_pretrained(model_id, cache_dir=cache_dir)
device = "cuda"

image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
def collate_fn(examples):
    texts = [example["prompt"] for example in examples]
    labels= [example['label'] for example in examples]
    images = [Image.open(example["image"]).convert("RGB") for example in examples]
    tokens = processor(text=texts, images=images, suffix=labels,
                    return_tensors="pt", padding="longest",
                    tokenize_newline_separately=False)
    
    tokens = tokens.to(torch.bfloat16).to(device)
    #tokens['labels'] = tokens['labels'].to(torch.bfloat16)
    return tokens




bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
#"""
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, quantization_config=bnb_config, 
                                                          torch_dtype=torch.bfloat16,
                                                          #revision="bfloat16",
                                                          device_map={"":0}, cache_dir=cache_dir) #{"":0}
                                                          
if SKIP_TRAIN:
    model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    model = model.merge_and_unload()
    #model.image_token_id = processor.image_token_id
    #model.config.pad_token_id = processor.pad_token_id

else:
    run_name = f"dataset={args.dataset}|dir={OUTPUT_DIR}|batch={args.batch}|lr={args.lr}"
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    wandb.init(project="finetune-vlm", name=run_name)
    args = TrainingArguments(
        num_train_epochs=3,
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=4,
        warmup_ratio=0.02,
        learning_rate=args.lr,
        #weight_decay=1e-6,
        #adam_beta2=0.999,
        logging_steps=1,
        max_grad_norm=0.3,
        optim="adamw_torch_fused",
        save_strategy="steps",
        save_steps=0.2,
        evaluation_strategy="steps",
        eval_steps=0.1,
        group_by_length=False,
        push_to_hub=False,
        #save_total_limit=1,
        output_dir=OUTPUT_DIR,
        bf16=True,
        #fp16=True,
        report_to="wandb",
        run_name=run_name,
        dataloader_pin_memory=False,
        lr_scheduler_type="cosine",
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn,
        args=args
    )

    trainer.train()

    trainer.save_model()

    model = trainer.model

model.eval()


def predict(ds):
    predictions =[]
    labels = []
    total = ds.shape[0]
    for i in tqdm(range(total)):
        try:
            image = Image.open(ds[i]["image"]).convert("RGB")
            prompt = ds[i]["prompt"]
            model_inputs = processor(text=prompt, images=image, return_tensors="pt")
            model_inputs = model_inputs.to(device)
            input_len = model_inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generation = model.generate(**model_inputs, max_new_tokens=150, do_sample=True, top_p=0.9, temperature=0.2)
                generation = generation[0][input_len:]
                response = processor.decode(generation, skip_special_tokens=True)
            response = response.lower()
            if "support" in response:
                predictions.append(label2id["supported"])
            elif "refute" in response:
                predictions.append(label2id["refuted"])
            elif "not enough info" in response or "nei" in response or "insufficient" in response:
                predictions.append(label2id["NEI"])
            else:
                print("Irrelevant response!!!")
                print(i, response)
                continue
            labels.append(label2id[ds[i]["label"]])
        except:
            continue
    return predictions, labels

predictions, targets = predict(dataset["test"])

print(confusion_matrix(targets, predictions))
print("f1-macro: ", f1_score(targets, predictions, average='macro'))
print("f1-micro: ", f1_score(targets, predictions, average='micro'))
print("f1-weighted: ", f1_score(targets, predictions, average='weighted'))
print("prec-score: ", precision_score(targets, predictions, average='weighted'))
print("recall-score: ", recall_score(targets, predictions, average='weighted'))
print("acc-score: ", accuracy_score(targets, predictions))