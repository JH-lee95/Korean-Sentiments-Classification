import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset,DataLoader

from transformers import TrainingArguments, Trainer
from transformers import BertForSequenceClassification, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import EarlyStoppingCallback

import gc


# tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-BERT-char16424")
# model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-BERT-char16424",num_labels=7)


# KR-BERT Load
# https://github.com/snunlp/KR-BERT
tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-Medium")
model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-Medium",num_labels=7)

data_path="/data/감성대화말뭉치_user1.csv"

# load and preprocess raw dataset
def preprocess_data(file_path,tokenizer=tokenizer):

    lbe=LabelEncoder()

    df=pd.read_csv(file_path)

    X=list(df['Sentence'])
    y=list(lbe.fit_transform(df['Emotion']))

    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.05,shuffle=True,stratify=y)
    X_train_tokenized=tokenizer(X_train,padding=True,truncation=True,max_length=128)
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=128)

    return X_train_tokenized,X_val_tokenized,y_train,y_val


#create torch dataset
class SentDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


# Define evaluation metrics
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall_micro = recall_score(y_true=labels, y_pred=pred,average="micro")
    recall_macro = recall_score(y_true=labels, y_pred=pred,average="macro")
    precision_micro = precision_score(y_true=labels, y_pred=pred,average="micro")
    precision_macro = precision_score(y_true=labels, y_pred=pred,average="macro")
    f1_macro = f1_score(y_true=labels, y_pred=pred,average="macro")

    return {"accuracy": accuracy, "recall_micro": recall_micro, "recall_macro": recall_macro, "precision_micro": precision_micro, 
                         "precision_macro": precision_macro ,"f1_macro": f1_macro}



def main():

    X_train_tokenized,X_val_tokenized,y_train,y_val=preprocess_data(data_path)

    train_dataset = SentDataset(X_train_tokenized, y_train)
    val_dataset = SentDataset(X_val_tokenized, y_val)


    args = TrainingArguments(
        save_total_limit=1,
        output_dir="/checkpoints/",
        evaluation_strategy="steps",
        #deepspeed="/home/deeptext/chatbot_jh/chatbot_ko_gpt_trininity/deepspeed_config.json",
        eval_steps=500,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=7,
        seed=42,
        load_best_model_at_end=True,
        learning_rate=5e-5 
    )

    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
   # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

def inference(model,text,device):

    model.eval()

    with torch.no_grad():

      encoding=tokenizer([text],padding=True,truncation=True,max_length=128,return_tensors="pt")
      outputs=model(**encoding.to(device))

      return text, outputs[0][0]

    
if __name__=="__main__":
    main()
