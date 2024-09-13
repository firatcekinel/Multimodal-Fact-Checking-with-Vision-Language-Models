from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, AutoModelForVision2Seq, AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor, ViTModel, AutoModel, SiglipVisionModel
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from transformers.image_utils import load_image
from transformers import BitsAndBytesConfig

system_prompt = "Assess the factuality of the following claim by considering evidence. You must only answer \"supported\", \"refuted\" or \"not enough info\".\n"

label2id = {"NEI":0, "not enough info":0, "supported":2, "refuted":1}
id2label = {0:"NEI", 0:"not enough info", 2:"supported", 1:"refuted"}

def filter_text(text, context_len=2500):
    words = text.split()  
    if len(words) > context_len:
        first_k_words = words[:context_len]  
        truncated_text = ' '.join(first_k_words) 
        return truncated_text
    return text
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

def get_image_url(dataset, row, idx, isclaim):
    if dataset == "mocheg":
        images = row["img_evidences"].split(";")
        return images[0]
    elif dataset == "factify2":
        if isclaim:
            return f"{idx}_claim.jpg"
        return f"{idx}_document.jpg"
    else:
        raise Exception("dataset not found!")

# Base class
class VLLM:
    def __init__(self, model_id="google/paligemma-3b-mix-448", task="multimodal", cache_dir=None, dataset="mocheg", isclaim=True):
        self.dataset = dataset 
        self.isclaim = isclaim

    @property
    def predict(self, df, img_path):
        raise NotImplementedError
        
    @property
    def getMMEmbeddings(self, df, img_path):
        raise NotImplementedError
    
    def getTextEmbeddings(self, df):
        text_embeddings = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            if self.isclaim:
                text = row.Claim
            else:
                #text = row.Evidence
                text = filter_text(row.Evidence, context_len=6000)
            inputs = self.tokenizer(text, return_tensors='pt')
            inputs = inputs.to(self.lm.device)
            with torch.inference_mode():
                outputs = self.lm(**inputs, output_hidden_states=True)
            lm_last_hidden_state = outputs.hidden_states[-1].mean(dim=1)
            text_embeddings.append(lm_last_hidden_state.detach().cpu().numpy())
        return text_embeddings
    
    def getImageEmbeddings(self, df, img_path):
        img_embeddings = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            url = img_path + get_image_url(dataset=self.dataset, row=row, idx=idx, isclaim=self.isclaim)
            image = Image.open(url).convert("RGB")
            inputs = self.processor(text="", images=image, padding="max_length", return_tensors="pt") #text=row.Claim, 
            inputs = inputs.to(self.vit.device)
            with torch.inference_mode():
                outputs = self.vit(**inputs, output_hidden_states=True)
            vit_last_hidden_state = outputs.vision_model_output.last_hidden_state.mean(dim=1)
            img_embeddings.append(vit_last_hidden_state.detach().cpu().numpy())
        return img_embeddings

    def text_predict(self, df):
        labels = []
        predictions = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            prompt = f"""{system_prompt}
Claim: {row.Claim}
Evidence: {row.Evidence}
"""#Evidence: {filter_text(row.Evidence, context_len=6000)}
            inputs = self.tokenizer(prompt, return_tensors='pt')
            inputs = inputs.to(self.lm.device)
            with torch.inference_mode():
                outputs = self.lm.generate(**inputs, max_new_tokens=50)
            
            response = self.tokenizer.decode(outputs[0]).lower()
            input_len = inputs["input_ids"].shape[-1]
            response = response[input_len:]

            if "not enough info" in response:
                predictions.append(label2id["NEI"])
            elif "refute" in response:
                predictions.append(label2id["refuted"])
            elif "support" in response:
                predictions.append(label2id["supported"])
            else:
                print("Irrelevant response!!!")
                print(idx, response)
                predictions.append(-1)
                #continue
            labels.append(label2id[row.cleaned_truthfulness])
        return predictions, labels

class QwenVL(VLLM):
    def __init__(self, model_id="Qwen/Qwen-VL-Chat-Int4", task="multimodal", cache_dir=None, dataset="mocheg", isclaim=True):
        self.dataset = dataset 
        self.isclaim = isclaim
        if task == "multimodal":
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", 
                                                              torch_dtype=torch.float16, trust_remote_code=True, cache_dir=cache_dir).eval()

        else:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat-Int4", trust_remote_code=True)
            self.lm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat-Int4", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, cache_dir=cache_dir).eval()

            self.processor = AutoImageProcessor.from_pretrained('laion/CLIP-ViT-bigG-14-laion2B-39B-b160k', cache_dir=cache_dir) #'google/vit-large-patch32-384'
            self.vit = ViTModel.from_pretrained('laion/CLIP-ViT-bigG-14-laion2B-39B-b160k', device_map="auto", torch_dtype=torch.float16, cache_dir=cache_dir).eval()

    def text_predict(self, df):
        return super().text_predict(df)

    def predict(self, df, img_path):
        predictions =[]
        labels = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            url = img_path + get_image_url(dataset=self.dataset, row=row, idx=idx, isclaim=self.isclaim)
            prompt = f""" {system_prompt}
Claim: {row.Claim}
Evidence: {row.Evidence}
"""#{filter_text(row.Evidence, context_len=6000)}
            query = self.tokenizer.from_list_format([
                {'image': url},
                {'text': prompt},
            ])
            response, history = self.model.chat(self.tokenizer, query=query, history=None)
            response = response.lower()
            if "not enough info" in response:
                predictions.append(label2id["NEI"])
            elif "refute" in response:
                predictions.append(label2id["refuted"])
            elif "support" in response:
                predictions.append(label2id["supported"])
            else:
                print("Irrelevant response!!!")
                print(idx, response)
                continue
            labels.append(label2id[row.cleaned_truthfulness])
        return predictions, labels

    def getMMEmbeddings(self, df, img_path):
        embeddings = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            url = img_path + get_image_url(dataset=self.dataset, row=row, idx=idx, isclaim=self.isclaim)
            if self.isclaim:
                text = row.Claim
            else:
                text = filter_text(row.Evidence, context_len=4000)

            query = self.tokenizer.from_list_format([
                {'image': url},
                {'text': text},
            ])

            inputs = self.tokenizer(query, return_tensors='pt')
            inputs = inputs.to(self.model.device)
            with torch.inference_mode():
                outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1].mean(dim=1)
            embeddings.append(last_hidden_state.detach().cpu().numpy())
        return embeddings
    
    def getTextEmbeddings(self, df):
        return super().getTextEmbeddings(df)

    def getImageEmbeddings(self, df, img_path):
        img_embeddings = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            url = img_path + get_image_url(dataset=self.dataset, row=row, idx=idx, isclaim=self.isclaim)
            image = Image.open(url).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = inputs.to(self.vit.device)
            with torch.no_grad():
                outputs = self.vit(**inputs)
            vit_last_hidden_state = outputs.last_hidden_state.mean(dim=1)
            img_embeddings.append(vit_last_hidden_state.detach().cpu().numpy())

        return img_embeddings
    
class Idefics2(VLLM):
    def __init__(self, model_id="HuggingFaceM4/idefics2-8b", task="multimodal", cache_dir=None, dataset="mocheg", isclaim=True):
        self.dataset = dataset
        self.isclaim = isclaim
        if task == "multimodal":
            self.tokenizer_mm = AutoProcessor.from_pretrained(model_id, do_image_splitting=False, size= {"longest_edge": 448, "shortest_edge": 378})
            self.model = AutoModelForVision2Seq.from_pretrained(model_id, cache_dir=cache_dir, torch_dtype=torch.float16, device_map="auto",quantization_config=quantization_config, _attn_implementation="flash_attention_2",).eval()
        
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", trust_remote_code=True)
            self.lm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map="auto", 
                                                           torch_dtype=torch.float16, trust_remote_code=True, cache_dir=cache_dir, 
                                                           quantization_config=quantization_config,_attn_implementation="flash_attention_2",
                                                           ).eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", trust_remote_code=True)
            self.lm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map="auto", 
                                                           torch_dtype=torch.float16, trust_remote_code=True, cache_dir=cache_dir, 
                                                           quantization_config=quantization_config,_attn_implementation="flash_attention_2",
                                                           ).eval()

            #"""
            self.processor = AutoProcessor.from_pretrained('google/siglip-so400m-patch14-384', cache_dir=cache_dir) 
            self.vit = AutoModel.from_pretrained('google/siglip-so400m-patch14-384', cache_dir=cache_dir)
            self.vit = self.vit.to("cuda")
            self.vit.eval()
            #"""
    def text_predict(self, df):
        return super().text_predict(df)

    def predict(self, df, img_path):
        predictions =[]
        labels = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            url = img_path + get_image_url(dataset=self.dataset, row=row, idx=idx, isclaim=self.isclaim)
            image = load_image(url)

            prompt = f""" {system_prompt}
Claim: {row.Claim}
Evidence: {row.Evidence}
"""
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ]
                },
            ]

            prompt = self.tokenizer_mm.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.tokenizer_mm(text=prompt, images=[image], return_tensors="pt") 
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=50)
                generated_texts = self.tokenizer_mm.batch_decode(generated_ids, skip_special_tokens=True)
            response_idx = generated_texts[0].find("\nAssistant:")
            response = generated_texts[0][response_idx:]
            response = response.lower()
            if "not enough info" in response:
                predictions.append(label2id["NEI"])
            elif "refute" in response:
                predictions.append(label2id["refuted"])
            elif "support" in response:
                predictions.append(label2id["supported"])
            else:
                print("Irrelevant response!!!")
                print(idx, response)
                predictions.append(-1)
                #continue

            labels.append(label2id[row.cleaned_truthfulness])
        return predictions, labels

    def getMMEmbeddings(self, df, img_path):
        embeddings = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            url = img_path + get_image_url(dataset=self.dataset, row=row, idx=idx, isclaim=self.isclaim)
            image = load_image(url)
            if self.isclaim:
                text = row.Claim
            else:
                text = row.Evidence#filter_text(row.Evidence, context_len=2500)
            messages = [
                {
                    "role": "user",
                    "content": [
                        #{"type": "image"},
                        #{"type": "text", "text": text},
                        {"type": "text", "text": row.Claim},
                        #{"type": "text", "text": filter_text(row.Evidence, context_len=6000)},
                    ]
                },
            ]
        
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.tokenizer(text=prompt, images=None, return_tensors="pt") #[image]
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1].mean(dim=1)
            embeddings.append(last_hidden_state.detach().cpu().numpy())
        return embeddings
            
    def getTextEmbeddings(self, df):
        text_embeddings = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            if self.isclaim:
                prompt = row.Claim
            else:
                prompt = row.Evidence#filter_text(row.Evidence, context_len=2500)
            text = "<s>[INST] " + prompt + " [/INST]" # 
            inputs = self.tokenizer(text, return_tensors='pt')
            inputs = inputs.to(self.lm.device)
            with torch.inference_mode():
                outputs = self.lm(**inputs, output_hidden_states=True)
            lm_last_hidden_state = outputs.hidden_states[-1].mean(dim=1)
            text_embeddings.append(lm_last_hidden_state.detach().cpu().numpy())
        return text_embeddings

    def getImageEmbeddings(self, df, img_path):
        return super().getImageEmbeddings(df, img_path)


class PaliGemma(VLLM):
    def __init__(self, model_id="google/paligemma-3b-mix-448", task="multimodal", cache_dir=None, dataset="mocheg", isclaim=True):
        self.dataset = dataset
        self.isclaim = isclaim
        if task == "multimodal":
            self.tokenizer = AutoProcessor.from_pretrained(model_id,  trust_remote_code=True)
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="auto", cache_dir=cache_dir,
                                                                           trust_remote_code=True, torch_dtype=torch.float16, 
                                                                           quantization_config=quantization_config,
                                                                           #_attn_implementation="flash_attention_2"
                                                                           )
            self.model.eval()
        else: # fusion
            self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", trust_remote_code=True)
            self.lm = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto", torch_dtype=torch.float16, 
                                                           trust_remote_code=True, cache_dir=cache_dir, 
                                                           quantization_config=quantization_config
                                                           ).eval()
            #"""
            self.processor = AutoProcessor.from_pretrained('google/siglip-so400m-patch14-384', cache_dir=cache_dir) 
            self.vit = AutoModel.from_pretrained('google/siglip-so400m-patch14-384', cache_dir=cache_dir)
            self.vit = self.vit.to("cuda")
            self.vit.eval()
            #"""
    def text_predict(self, df):
        return super().text_predict(df)

    def predict(self, df, img_path):
        predictions =[]
        labels = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            url = img_path + get_image_url(dataset=self.dataset, row=row, idx=idx, isclaim=self.isclaim)
            image = Image.open(url).convert('RGB')
            prompt = f""" {system_prompt}
Claim: {row.Claim}
Evidence: {row.Evidence}
"""
#Evidence: {filter_text(row.Evidence, context_len=5000)}
            model_inputs = self.tokenizer(text=prompt, images=image, return_tensors="pt")
            model_inputs = model_inputs.to("cuda")
            input_len = model_inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generation = self.model.generate(**model_inputs, max_new_tokens=50, do_sample=False)
                generation = generation[0][input_len:]
                response = self.tokenizer.decode(generation, skip_special_tokens=True)
            response = response.lower()
            if "not enough info" in response:
                predictions.append(label2id["NEI"])
            elif "refute" in response:
                predictions.append(label2id["refuted"])
            elif "support" in response:
                predictions.append(label2id["supported"])
            else:
                print("Irrelevant response!!!")
                print(idx, response)
                continue
            labels.append(label2id[row.cleaned_truthfulness])
        return predictions, labels

    def getMMEmbeddings(self, df, img_path):
        embeddings = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

            url = img_path + get_image_url(dataset=self.dataset, row=row, idx=idx, isclaim=self.isclaim)
            image = Image.open(url).convert('RGB')
            if self.isclaim:
                prompt = row.Claim
            else:
                prompt = row.Claim + "\n" + filter_text(row.Evidence, context_len=5000)

            prompt = ""
            model_inputs = self.tokenizer(text=prompt, images=image, return_tensors="pt")
            model_inputs = model_inputs.to("cuda")
            with torch.inference_mode():
                outputs = self.model(**model_inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1].mean(dim=1)
            embeddings.append(last_hidden_state.detach().cpu().numpy())
        return embeddings
    def getTextEmbeddings(self, df):
        return super().getTextEmbeddings(df)
    def getImageEmbeddings(self, df, img_path):
        return super().getImageEmbeddings(df, img_path)
