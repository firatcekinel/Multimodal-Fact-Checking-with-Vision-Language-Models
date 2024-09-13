import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import wandb
import argparse

from vlm import QwenVL, Idefics2, MiniCPMV, PaliGemma
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

label2id = {"NEI":0, "supported":2, "refuted":1, 
            "Support_Multimodal":3, "Support_Text":4, "Insufficient_Multimodal":0, "Insufficient_Text":1, "Refute":2}
id2label = {0:"NEI", 2:"supported", 1:"refuted",
            3:"Support_Multimodal", 4:"Support_Text", 0:"Insufficient_Multimodal", 1:"Insufficient_Text", 2:"Refute"}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pl.seed_everything(42)

class FusionDataset(Dataset):
    def __init__(self, data, embeddings):
        self.data = data
        self.embeddings = embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        return dict(
            label=label2id[row.cleaned_truthfulness],
            text=row.Claim,
            embedding=torch.tensor(self.embeddings[idx]).reshape(-1),
        )

class FusionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_embeddings,
        val_embeddings,
        test_embeddings,
        batch_size: int = 1,
    ):
        super().__init__()

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size

        self.train_embeddings = train_embeddings
        self.val_embeddings = val_embeddings
        self.test_embeddings = test_embeddings

    def setup(self, stage=None):
        self.train_dataset = FusionDataset(
            self.train_df,
            self.train_embeddings,
        )
        self.val_dataset = FusionDataset(
            self.val_df,
            self.val_embeddings,
        )
        self.test_dataset = FusionDataset(
            self.test_df,
            self.test_embeddings,
        )
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

class FusionModel(pl.LightningModule):
    def __init__(self, input_size1=4096, hidden_size=256, lr=0.01, dropout=0.1, num_classes=3, training_steps=None, warmup_steps=None, class_weights=torch.ones(3).to(device)):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(input_size1)
        self.embedding1_fc = torch.nn.Linear(input_size1, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size//8)
        #self.fc1 = torch.nn.Linear(hidden_size, num_classes)
        self.fc2= torch.nn.Linear(hidden_size//8, num_classes)
        self.dropout = torch.nn.Dropout(dropout) 
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)#

        self.training_steps = training_steps
        self.warmup_steps = warmup_steps
        self.lr = lr

    def forward(self, emb1):
        out1 = self.dropout(self.relu(self.embedding1_fc(emb1)))
        output = self.dropout(self.relu(self.fc1(out1)))
        #output = self.fc1(out1)
        output = self.fc2(output)
        #output = torch.sigmoid(output)
        return output

    def training_step(self, batch, batch_size):
        label = batch["label"]
        embedding = batch["embedding"]

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self(embedding)
            loss = self.criterion(outputs, label)

		
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "labels": batch["label"], "preds": outputs}
    
    def validation_step(self, batch, batch_size):
        label = batch["label"]
        embedding = batch["embedding"]

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self(embedding)
            loss = self.criterion(outputs, label)
        
        self.log("val_loss", loss, prog_bar=True, logger=True)

        return {"loss": loss, "labels": batch["label"], "preds": outputs}
    
    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=self.lr)

        scheduler = get_cosine_schedule_with_warmup(
          optimizer,
          num_warmup_steps=self.warmup_steps,
          num_training_steps=self.training_steps
        )

        return dict(
          optimizer=optimizer,
          lr_scheduler=dict(
            scheduler=scheduler,
            interval='step'
          )
        )
    
def init_model(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config, project=wandb_proj_name) as run:
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        params = {
            "batch_size": config.batch_size, 
            "lr": config.learning_rate, 
            "hidden_size":config.hidden_size,
            "dropout":config.dropout
            }
        
        print(f"Run name: {str(params)}")
        run_name = f"""model:{model_name}|data:{args.dataset}|batch:{str(params["batch_size"])}|hidden_size:{str(params["hidden_size"])}|lr:{str(params["lr"])}|drop:{str(params["dropout"])}|mm_emb:{str(input_size1)}""" 
        params["run_name"] = run_name
        run.name = run_name
        trainModel(params)

def trainModel(params):
    global result_dict

    steps_per_epoch=len(train_df) // params["batch_size"]
    total_training_steps = steps_per_epoch * epochs
    warmup_steps = total_training_steps // 20

    data_module = FusionDataModule(train_df, val_df, test_df, 
                               embeddings1, embeddings2, embeddings3,
                               params["batch_size"])

    model = FusionModel(input_size1, params["hidden_size"], params["lr"], params["dropout"], num_classes, warmup_steps, total_training_steps, class_weights)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.dataset + '/outputs/checkpoints-'+ args.wandb_proj_name + '-sweep-'+model_name,
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
    progress_bar_callback = TQDMProgressBar(refresh_rate=20)

    logger = WandbLogger(project=wandb_proj_name, name=params["run_name"])

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback, progress_bar_callback],
        max_epochs=epochs,
        accelerator="gpu",
    )

    trainer.fit(model, data_module)
    model = FusionModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, input_size1=input_size1, hidden_size=params["hidden_size"], dropout=params["dropout"], num_classes=num_classes, class_weights=class_weights)

    model.freeze()
    model.eval()

    predictions = []
    targets = []

    for batch in data_module.test_dataloader():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(batch["embedding"].to(device))

        y_preds = torch.max(outputs, axis=1).indices.detach().cpu().tolist()
        y_true = batch["label"].detach().cpu().tolist()

        predictions = predictions + y_preds
        targets = targets + y_true

    conf_matrix = str(confusion_matrix(targets, predictions).tolist())
    metrics = {
        "f1-macro" : f1_score(targets, predictions, average='macro'),
        "f1-micro" : f1_score(targets, predictions, average='micro'),
        "f1-weighted" : f1_score(targets, predictions, average='weighted'),
        "precision" : precision_score(targets, predictions, average='weighted'),
        "recall" : recall_score(targets, predictions, average='weighted'),
        "accuracy" : accuracy_score(targets, predictions)
    }

    result_dict["confusion matrix"].append(conf_matrix)
    result_dict["metrics"].append(metrics)
    result_dict["run name"].append(str(params))
    result_dict["f1-macro"].append(metrics["f1-macro"])

    print(confusion_matrix(targets, predictions))
    print(f"""f1-macro: {f1_score(targets, predictions, average='macro')}, f1-micro: {f1_score(targets, predictions, average='micro')}, f1-weighted: {f1_score(targets, predictions, average='weighted')}, prec-score: {precision_score(targets, predictions, average='weighted')}, recall-score: {recall_score(targets, predictions, average='weighted')}, acc-score: {accuracy_score(targets, predictions)}""")


import os.path


def filter_data(df, dir_path):
    claim_img = []
    doc_img = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        if os.path.exists(f"{dir_path}/{idx}_claim.jpg"):
            claim_img.append(True)
        else:
            claim_img.append(None)
        
        if os.path.exists(f"{dir_path}/{idx}_document.jpg"):
            doc_img.append(True)
        else:
            doc_img.append(None)

    df["claim_img_exists"] = claim_img
    df["doc_img_exists"] = doc_img

    return df

def get_model(model_name):
    model_name = model_name.lower()
    if "idefics" in model_name:
        return Idefics2(cache_dir=cache_dir, task="multimodal", dataset=args.dataset, isclaim=args.isclaim)
    elif "qwen" in model_name:
        return QwenVL(cache_dir=cache_dir, task="multimodal", dataset=args.dataset, isclaim=args.isclaim)
    if "gemma" in model_name:
        return PaliGemma(cache_dir=cache_dir, task="multimodal", dataset=args.dataset, isclaim=args.isclaim)
    if "minicpm" in model_name:
        return MiniCPMV(cache_dir=cache_dir, task="multimodal", dataset=args.dataset, isclaim=args.isclaim)

parser = argparse.ArgumentParser() 
parser.add_argument('--model_id', default='google/paligemma-3b', type=str, help='model id')
parser.add_argument('--dataset', default='mocheg', type=str, help='dataset name')
parser.add_argument('--isclaim', default=True, type=bool, help='claim or evidence')
parser.add_argument('--train_embedding_file', default='train_claim_embeddings.npy', type=str, help='train_embedding_file')
parser.add_argument('--val_embedding_file', default='val_claim_embeddings.npy', type=str, help='val_embedding_file')
parser.add_argument('--test_embedding_file', default='test_claim_embeddings.npy', type=str, help='test_embedding_file')
parser.add_argument('--wandb_proj_name', default='mm_sweep3', type=str, help='wandb proj name')
parser.add_argument('--cache_dir', default=None, type=str, help='hf model cache dir')
args = parser.parse_args()

print(f"{args.model_id}|{args.dataset}|{args.wandb_proj_name}")

predictions = []
targets = []
num_classes = 3
epochs = 50
wandb_proj_name = args.wandb_proj_name
model_id = args.model_id # selected model
model_name = model_id.split("/")[-1]
embedding_path = f"{args.dataset}/embeddings/{model_name}/multimodal/" # path of embeddings
print(f"embedding path: {embedding_path}")
saveEmbeddings = False # always False to train the model
make_prediction = False

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

elif args.dataset == "factify2":
    train_df = pd.read_csv(f"{args.dataset}/train.csv", encoding="utf-8", sep="\t")
    train_df = filter_data(train_df, dir_path=f"{args.dataset}/images/train")
    train_df = train_df[["claim", "claim_image", "document", "document_image", "claim_img_exists", "doc_img_exists", "Category"]]
    train_df = train_df.rename(columns={'claim': 'Claim', 'document': 'Evidence', 'Category' : 'cleaned_truthfulness'})
    train_df.dropna(inplace=True)

    val_df = pd.read_csv(f"{args.dataset}/val.csv", encoding="utf-8", sep="\t")
    val_df = filter_data(val_df, dir_path=f"{args.dataset}/images/val")
    val_df = val_df[["claim", "claim_image", "document", "document_image", "claim_img_exists", "doc_img_exists", "Category"]]
    val_df = val_df.rename(columns={'claim': 'Claim', 'document': 'Evidence', 'Category' : 'cleaned_truthfulness' })
    val_df.dropna(inplace=True)
    
    test_df = pd.read_csv(f"{args.dataset}/test.csv", encoding="utf-8", sep="\t")
    test_df = filter_data(test_df, dir_path=f"{args.dataset}/images/test")
    test_df = test_df[["claim", "claim_image", "document", "document_image", "claim_img_exists", "doc_img_exists"]]
    test_df = test_df.rename(columns={'claim': 'Claim', 'document': 'Evidence'})
    test_df.dropna(inplace=True)
    
result_dict = {"run name":[], "metrics": [], "confusion matrix": [], "f1-macro":[]}

if saveEmbeddings: # ignore this part
    cache_dir = args.cache_dir
    model = get_model(model_name)

    if args.dataset == "mocheg":
        embeddings1 = model.getMMEmbeddings(train_df, f"../multimodal-fc/{args.dataset}/train/images/")
        np.save(embedding_path + args.train_embedding_file, np.array(embeddings1))
        embeddings2 = model.getMMEmbeddings(val_df, f"../multimodal-fc/{args.dataset}/val/images/")
        np.save(embedding_path + args.val_embedding_file, np.array(embeddings2))
        embeddings3 = model.getMMEmbeddings(test_df, f"../multimodal-fc/{args.dataset}/test/images/")
        np.save(embedding_path + args.test_embedding_file, np.array(embeddings3))

    elif args.dataset == "factify2":
        embeddings1 = model.getMMEmbeddings(train_df, f"{args.dataset}/images/train/")
        np.save(embedding_path + args.train_embedding_file, np.array(embeddings1))

        embeddings2 = model.getMMEmbeddings(val_df, f"{args.dataset}/images/val/")
        np.save(embedding_path + args.val_embedding_file, np.array(embeddings2))

        embeddings3 = model.getMMEmbeddings(test_df, f"{args.dataset}/images/test/")
        np.save(embedding_path + args.test_embedding_file, np.array(embeddings3))
    torch.cuda.empty_cache()

elif make_prediction:
    cache_dir = args.cache_dir
    model = get_model(model_name)
    if args.dataset == "mocheg":
        predictions, targets = model.predict(test_df, f"../multimodal-fc/{args.dataset}/test/images/")
    elif args.dataset == "factify2":
        # Define a dictionary for replacements
        mapping = {'Support_Multimodal': 'supported',
                'Support_Text': 'supported',
                    'Refute': 'refuted',
                    'Insufficient_Multimodal': 'NEI',
                    'Insufficient_Text': 'NEI'}

        # Rename values based on the dictionary
        val_df['cleaned_truthfulness'] = val_df['cleaned_truthfulness'].replace(mapping, regex=True)
        predictions, targets = model.predict(val_df, f"{args.dataset}/images/val/")

    
    print(confusion_matrix(targets, predictions))
    print("f1-macro: ", f1_score(targets, predictions, average='macro'))
    print("f1-micro: ", f1_score(targets, predictions, average='micro'))
    print("f1-weighted: ", f1_score(targets, predictions, average='weighted'))
    print("prec-score: ", precision_score(targets, predictions, average='weighted'))
    print("recall-score: ", recall_score(targets, predictions, average='weighted'))
    print("acc-score: ", accuracy_score(targets, predictions))
else:
    if args.dataset == "factify2":
        # Define a dictionary for replacements
        mapping = {'Support_Multimodal': 'supported',
                'Support_Text': 'supported',
                    'Refute': 'refuted',
                    'Insufficient_Multimodal': 'NEI',
                    'Insufficient_Text': 'NEI'}

        # Rename values based on the dictionary
        train_df['cleaned_truthfulness'] = train_df['cleaned_truthfulness'].replace(mapping, regex=True)
        val_df['cleaned_truthfulness'] = val_df['cleaned_truthfulness'].replace(mapping, regex=True)

        train_df.reset_index(inplace=True)
        val_df.reset_index(inplace=True)
        test_df = val_df.copy()
        x = train_df.drop(columns=['cleaned_truthfulness'])  # Features
        y = train_df['cleaned_truthfulness']  # Target
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_test, y_test], axis=1)
        indices = val_df.index.to_list()
        print(f"Train-val-test sizes: {train_df.shape[0]}, {val_df.shape[0]}, {test_df.shape[0]}")


    # class weights for cross entropy loss
    class_counts = train_df.cleaned_truthfulness.value_counts().sort_index().values
    total_samples = train_df.shape[0]
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.from_numpy(class_weights).float().to(device)
    print("class_weights: ", class_weights)

    # Load embeddings
    embeddings1 = np.load(embedding_path + args.train_embedding_file)
    embeddings2 = np.load(embedding_path + args.val_embedding_file)
    embeddings3 = np.load(embedding_path + args.test_embedding_file)

    if args.dataset == "factify2": # test set has no labels, use val set for testing
        embeddings3 = embeddings2
        embeddings2 = embeddings1[indices]
        mask = np.ones(len(embeddings1), dtype=bool)
        mask[indices] = False
        embeddings1 = embeddings1[mask]

        #train_df.to_csv(f"Multimodal-Fact-Checking/{args.dataset}/train_baseline.csv", sep="\t")
        #val_df.to_csv(f"Multimodal-Fact-Checking/{args.dataset}/val_baseline.csv", sep="\t")
        #test_df.to_csv(f"Multimodal-Fact-Checking/{args.dataset}/test_baseline.csv", sep="\t")

        #np.save(embedding_path + "train_text_baseline.npy", np.array(embeddings1))
        #np.save(embedding_path + "val_text_baseline.npy", np.array(embeddings2))
        #np.save(embedding_path + "test_text_baseline.npy", np.array(embeddings3))

    input_size1 = embeddings1[0].shape[-1]

    sweep_config = {
        'method': 'grid', 
        'metric': {
        'name': 'val_loss',
        'goal': 'minimize'   
        },
        'parameters': {
            'learning_rate': {
                'values': [0.000001, 0.00001, 0.0001, 0.001, 0.01]
                },
            'batch_size': {
                'values': [32, 64, 128]
                },
            'hidden_size': {
                'values': [128, 256, 512]
                },
            'dropout': {
                'values': [0.05, 0.1, 0.2]
                },
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=wandb_proj_name)

    wandb.agent(sweep_id, function=init_model, count=135)

    result_df = pd.DataFrame.from_dict(result_dict)
    output_file_name = f"sweep_mm_{model_name}_{args.wandb_proj_name}"
    result_df.to_csv(embedding_path + output_file_name+ ".csv", sep="\t", encoding="utf-8")