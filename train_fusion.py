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

label2id = {"NEI":0, "supported":2, "refuted":1, 
            "Support_Multimodal":3, "Support_Text":4, "Insufficient_Multimodal":0, "Insufficient_Text":1, "Refute":2}
id2label = {0:"NEI", 2:"supported", 1:"refuted",
            3:"Support_Multimodal", 4:"Support_Text", 0:"Insufficient_Multimodal", 1:"Insufficient_Text", 2:"Refute"}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pl.seed_everything(42)

class FusionDataset(Dataset):
    def __init__(self, data, text_embeddings, image_embeddings):
        self.data = data
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        return dict(
            label=label2id[row.cleaned_truthfulness],
            text=row.Claim,
            text_embedding=torch.tensor(self.text_embeddings[idx]).reshape(-1),
            image_embedding=torch.tensor(self.image_embeddings[idx]).reshape(-1),
        )

class FusionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_text_embeddings,
        train_image_embeddings,
        val_text_embeddings,
        val_image_embeddings,
        test_text_embeddings,
        test_image_embeddings,
        batch_size: int = 1,
    ):
        super().__init__()

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size

        self.train_text_embeddings = train_text_embeddings
        self.train_image_embeddings = train_image_embeddings
        self.val_text_embeddings = val_text_embeddings
        self.val_image_embeddings = val_image_embeddings
        self.test_text_embeddings = test_text_embeddings
        self.test_image_embeddings = test_image_embeddings

    def setup(self, stage=None):
        self.train_dataset = FusionDataset(
            self.train_df,
            self.train_text_embeddings,
            self.train_image_embeddings
        )
        self.val_dataset = FusionDataset(
            self.val_df,
            self.val_text_embeddings,
            self.val_image_embeddings
        )
        self.test_dataset = FusionDataset(
            self.test_df,
            self.test_text_embeddings,
            self.test_image_embeddings
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
    def __init__(self, input_size1=1024, input_size2=4096, hidden_size=256, lr=0.01, dropout=0.1, num_classes=3, training_steps=None, warmup_steps=None, class_weights=torch.ones(3).to(device)):
        super().__init__()
        self.embedding1_fc = torch.nn.Linear(input_size1, hidden_size)
        self.embedding2_fc = torch.nn.Linear(input_size2, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(hidden_size*2, hidden_size//4)
        #self.fc1 = torch.nn.Linear(hidden_size*2, num_classes)
        self.fc2= torch.nn.Linear(hidden_size//4, num_classes)
        self.dropout = torch.nn.Dropout(dropout) 
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        self.training_steps = training_steps
        self.warmup_steps = warmup_steps
        self.lr=lr

    def forward(self, emb1, emb2):
        out1 = self.dropout(self.relu(self.embedding1_fc(emb1)))
        out2 = self.dropout(self.relu(self.embedding2_fc(emb2)))
        combined = torch.cat((out1, out2), dim=1)
        output = self.dropout(self.relu(self.fc1(combined)))
        #output = self.fc1(combined)
        output = self.fc2(output)
        #output = torch.softmax(output)
        return output

    def training_step(self, batch, batch_size):
        label = batch["label"]
        text_embedding = batch["text_embedding"]
        image_embedding = batch["image_embedding"]

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self(image_embedding, text_embedding)
            loss = self.criterion(outputs, label)

		
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "labels": batch["label"], "preds": outputs}
    
    def validation_step(self, batch, batch_size):
        label = batch["label"]
        text_embedding = batch["text_embedding"]
        image_embedding = batch["image_embedding"]

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self(image_embedding, text_embedding)
            loss = self.criterion(outputs, label)
        
        self.log("val_loss", loss, prog_bar=True, logger=True)

        return {"loss": loss, "labels": batch["label"], "preds": outputs}

    def test_step(self, batch, batch_size):
        global predictions
        global targets
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self(batch["image_embedding"], batch["text_embedding"])

        y_preds = torch.max(outputs, axis=1).indices.detach().cpu().tolist()
        y_true = batch["label"].detach().cpu().tolist()

        predictions = predictions + y_preds
        targets = targets + y_true
    
    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=self.lr)

        scheduler = get_linear_schedule_with_warmup(
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
        run_name = f"""model:{model_name}|data:{args.dataset}|batch:{str(params["batch_size"])}|hidden_size:{str(params["hidden_size"])}|lr:{str(params["lr"])}|drop:{str(params["dropout"])}|img_emb:{str(input_size1)}|txt_emb:{str(input_size2)}"""
        params["run_name"] = run_name
        run.name = run_name
        print(params)
        trainModel(params)

def trainModel(params):
    global result_dict
    data_module = FusionDataModule(train_df, val_df, test_df, 
                               text_embeddings1, img_embeddings1,
                               text_embeddings2, img_embeddings2,
                               text_embeddings3, img_embeddings3,
                               params["batch_size"])

    steps_per_epoch=len(train_df) // params["batch_size"]
    total_training_steps = steps_per_epoch * epochs
    warmup_steps = total_training_steps // 20

    model = FusionModel(input_size1, input_size2, params["hidden_size"], params["lr"], params["dropout"], num_classes, warmup_steps, total_training_steps, class_weights)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.dataset + '/outputs/checkpoints-'+ args.wandb_proj_name + '-sweep-'+model_name,
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
    progress_bar_callback = TQDMProgressBar(refresh_rate=30)

    project_name = f"""data:{args.dataset}|batch:{str(params["batch_size"])}|hidden_size:{str(params["hidden_size"])}|lr:{str(params["lr"])}|text_emb:{str(input_size2)}|image_emb:{str(input_size1)}"""
    logger = WandbLogger(project=wandb_proj_name, name=project_name)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback, progress_bar_callback],
        max_epochs=epochs,
        accelerator="gpu",
    )

    trainer.fit(model, data_module)

    model = FusionModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                             input_size1=input_size1, input_size2=input_size2, 
                                             hidden_size=params["hidden_size"], dropout=params["dropout"], num_classes=num_classes, class_weights=class_weights)

    model.freeze()
    model.eval()

    predictions = []
    targets = []
    
    for batch in data_module.test_dataloader():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(batch["image_embedding"].to(device), batch["text_embedding"].to(device))

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
        return Idefics2(cache_dir=cache_dir, task="fusion", dataset=args.dataset, isclaim=args.isclaim)
    elif "qwen" in model_name:
        return QwenVL(cache_dir=cache_dir, task="fusion", dataset=args.dataset, isclaim=args.isclaim)
    if "gemma" in model_name:
        return PaliGemma(cache_dir=cache_dir, task="fusion", dataset=args.dataset, isclaim=args.isclaim)
    if "minicpm" in model_name:
        return MiniCPMV(cache_dir=cache_dir, task="fusion", dataset=args.dataset, isclaim=args.isclaim)

parser = argparse.ArgumentParser() 
parser.add_argument('--model_id', default='Qwen/Qwen-VL-Chat-Int4', type=str, help='model id')
parser.add_argument('--dataset', default='mocheg', type=str, help='dataset name')
parser.add_argument('--isclaim', default=True, type=bool, help='claim or evidence')
parser.add_argument('--train_image_file', default='multimodal/train_claim_embeddings.npy', type=str, help='train_image_file')
parser.add_argument('--val_image_file', default='multimodal/val_claim_embeddings.npy', type=str, help='val_image_file')
parser.add_argument('--test_image_file', default='multimodal/test_claim_embeddings.npy', type=str, help='test_image_file')
parser.add_argument('--train_text_file', default='multimodal/train_text_embeddings.npy', type=str, help='train_text_file')
parser.add_argument('--val_text_file', default='multimodal/val_text_embeddings.npy', type=str, help='val_text_file')
parser.add_argument('--test_text_file', default='multimodal/test_text_embeddings.npy', type=str, help='test_text_file')
parser.add_argument('--wandb_proj_name', default='mmev_sweep3', type=str, help='wandb proj name')
#parser.add_argument('--wandb_proj_name', default='fusion3_claim+image', type=str, help='wandb proj name')
parser.add_argument('--cache_dir', default=None, type=str, help='hf model cache dir')
args = parser.parse_args()

print(f"{args.model_id}|{args.dataset}|{args.wandb_proj_name}")

input_size1 = 768 # image embedding
input_size2 = 4096 # text embedding
#hidden_size = 256
num_classes = 3
#batch_size = 64
epochs = 50
wandb_proj_name= args.wandb_proj_name
model_id = args.model_id # selected model
model_name = model_id.split("/")[-1]
embedding_path = f"{args.dataset}/embeddings/{model_name}/"
saveEmbeddings = False # always False to train the model

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
    test_df = filter_data(test_df, dir_path="factify2/images/test")
    test_df = test_df[["claim", "claim_image", "document", "document_image", "claim_img_exists", "doc_img_exists"]]
    test_df = test_df.rename(columns={'claim': 'Claim', 'document': 'Evidence'})
    test_df.dropna(inplace=True)


result_dict = {"run name":[], "metrics": [], "confusion matrix": [], "f1-macro":[]}

if saveEmbeddings: # ignore this part
    cache_dir = args.cache_dir

    model = get_model(model_name)
    #"""
    # Text embeddings
    embeddings1 = model.getTextEmbeddings(train_df)
    np.save(embedding_path + args.train_text_file, np.array(embeddings1))

    embeddings2 = model.getTextEmbeddings(val_df)
    np.save(embedding_path + args.val_text_file, np.array(embeddings2))

    embeddings3 = model.getTextEmbeddings(test_df)
    np.save(embedding_path + args.test_text_file, np.array(embeddings3))
    #"""
    #torch.cuda.empty_cache()
    # Image embeddings
    if args.dataset == "mocheg":
        embeddings1 = model.getImageEmbeddings(train_df, f"../multimodal-fc/{args.dataset}/train/images/")
        np.save(embedding_path + args.train_image_file, np.array(embeddings1))
        embeddings2 = model.getImageEmbeddings(val_df, f"../multimodal-fc/{args.dataset}/val/images/")
        np.save(embedding_path + args.val_image_file, np.array(embeddings2))
        embeddings3 = model.getImageEmbeddings(test_df, f"../multimodal-fc/{args.dataset}/test/images/")
        np.save(embedding_path + args.test_image_file, np.array(embeddings3))
    elif args.dataset == "factify2":
        embeddings1 = model.getImageEmbeddings(train_df, f"{args.dataset}/images/train/")
        np.save(embedding_path + args.train_image_file, np.array(embeddings1))
        embeddings2 = model.getImageEmbeddings(val_df, f"{args.dataset}/images/val/")
        np.save(embedding_path + args.val_image_file, np.array(embeddings2))
        embeddings3 = model.getImageEmbeddings(test_df, f"{args.dataset}/images/test/")
        np.save(embedding_path + args.test_image_file, np.array(embeddings3))
    #"""

else:
    if "factify2" in args.dataset:
        # Define a dictionary for replacements
        mapping = {'Support_Multimodal': 'supported',
                'Support_Text': 'supported',
                    'Refute': 'refuted',
                    'Insufficient_Multimodal': 'NEI',
                    'Insufficient_Text': 'NEI'}

        # Rename values based on the dictionary
        train_df['cleaned_truthfulness'] = train_df['cleaned_truthfulness'].replace(mapping, regex=True)
        val_df['cleaned_truthfulness'] = val_df['cleaned_truthfulness'].replace(mapping, regex=True)
        #test_df['cleaned_truthfulness'] = test_df['cleaned_truthfulness'].replace(mapping, regex=True)


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
    text_embeddings1 = np.load(embedding_path + args.train_text_file)
    text_embeddings2 = np.load(embedding_path + args.val_text_file)
    text_embeddings3 = np.load(embedding_path + args.test_text_file)

    img_embeddings1 = np.load(embedding_path + args.train_image_file)
    img_embeddings2 = np.load(embedding_path + args.val_image_file)
    img_embeddings3 = np.load(embedding_path + args.test_image_file)

    if args.dataset == "factify2": # test set has no labels, use val set for testing
        text_embeddings3 = text_embeddings2
        text_embeddings2 = text_embeddings1[indices]
        mask = np.ones(len(text_embeddings1), dtype=bool)
        mask[indices] = False
        text_embeddings1 = text_embeddings1[mask]

        img_embeddings3 = img_embeddings2
        img_embeddings2 = img_embeddings1[indices]
        mask = np.ones(len(img_embeddings1), dtype=bool)
        mask[indices] = False
        img_embeddings1 = img_embeddings1[mask]

    input_size1 = img_embeddings1[0].shape[-1] # image embedding
    input_size2 = text_embeddings1[0].shape[-1] # text embedding

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
    #  wandb.login(key=WANDB_API_KEY)
    sweep_id = wandb.sweep(sweep_config, project=wandb_proj_name)

    wandb.agent(sweep_id, function=init_model, count=135)

    result_df = pd.DataFrame.from_dict(result_dict)
    output_file_name = f"sweep_{model_name}_{args.wandb_proj_name}"
    result_df.to_csv(embedding_path + output_file_name + ".csv", sep="\t", encoding="utf-8")