from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
import argparse
import os
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser() 
parser.add_argument('--model_id', default='Qwen/Qwen-VL-Chat-Int4', type=str, help='model id')
parser.add_argument('--dataset', default='mocheg', type=str, help='dataset name')
parser.add_argument('--classifier', default='svm', type=str, help='classifier name')
parser.add_argument('--useEvidence', default=True, type=bool, help='use evidence emb')
parser.add_argument('--train_embedding_file', default='train_claim_embeddings.npy', type=str, help='train_embedding_file')
parser.add_argument('--test_embedding_file', default='test_claim_embeddings.npy', type=str, help='test_embedding_file')
parser.add_argument('--train_evidence_file', default='train_text_embeddings.npy', type=str, help='train_evidence_file')
parser.add_argument('--test_evidence_file', default='test_text_embeddings.npy', type=str, help='test_evidence_file')
args = parser.parse_args()

print(f"{args.model_id}|{args.dataset}|{args.classifier}|evidence:{args.useEvidence}")

model_id = args.model_id # selected model
model_name = model_id.split("/")[-1]
embedding_path = f"{args.dataset}/embeddings/{model_name}/multimodal/"


if args.dataset == "mocheg":
    train_df = pd.read_csv(f"{args.dataset}/train.csv", encoding="utf-8")
    train_df = train_df[["Claim", "Evidence", "img_evidences", "cleaned_truthfulness"]]
    train_df.dropna(inplace=True)

    test_df = pd.read_csv(f"{args.dataset}/test.csv", encoding="utf-8")
    test_df = test_df[["Claim", "Evidence", "img_evidences", "cleaned_truthfulness"]]
    test_df.dropna(inplace=True)

elif args.dataset == "factify2":
    train_df = pd.read_csv(f"{args.dataset}/train_baseline.csv", encoding="utf-8", sep="\t")
    train_df.dropna(inplace=True)
    
    test_df = pd.read_csv(f"{args.dataset}/test_baseline.csv", encoding="utf-8", sep="\t")
    test_df.dropna(inplace=True)
    """
    mapping = {'Support_Multimodal': 'supported',
                'Support_Text': 'supported',
                    'Refute': 'refuted',
                    'Insufficient_Multimodal': 'NEI',
                    'Insufficient_Text': 'NEI'}

    # Rename values based on the dictionary
    train_df['cleaned_truthfulness'] = train_df['cleaned_truthfulness'].replace(mapping, regex=True)
    test_df['cleaned_truthfulness'] = test_df['cleaned_truthfulness'].replace(mapping, regex=True)
    """
#"Support_Multimodal":3, "Support_Text":4, "Insufficient_Multimodal":0, "Insufficient_Text":1, "Refute":2
train_df["cleaned_truthfulness"] = train_df["cleaned_truthfulness"].replace({'supported': 2, 'refuted': 1, 'NEI': 0})
#train_df["cleaned_truthfulness"] = train_df["cleaned_truthfulness"].replace({
#    "Support_Multimodal":3, "Support_Text":4, "Insufficient_Multimodal":0, "Insufficient_Text":1, "Refute":2})
y_train = train_df["cleaned_truthfulness"].to_list()
y_train = np.array(y_train)

test_df["cleaned_truthfulness"] = test_df["cleaned_truthfulness"].replace({'supported': 2, 'refuted': 1, 'NEI': 0})
#test_df["cleaned_truthfulness"] = test_df["cleaned_truthfulness"].replace({
#    "Support_Multimodal":3, "Support_Text":4, "Insufficient_Multimodal":0, "Insufficient_Text":1, "Refute":2})
y_test = test_df["cleaned_truthfulness"].to_list()
y_test = np.array(y_test)

embeddings1 = np.load(embedding_path + args.train_embedding_file)
embeddings3 = np.load(embedding_path + args.test_embedding_file)

if args.useEvidence:
    embeddings11 = np.load(embedding_path + args.train_evidence_file)
    embeddings33 = np.load(embedding_path + args.test_evidence_file)

    embeddings1 = np.concatenate((embeddings1, embeddings11), axis=2)
    embeddings3 = np.concatenate((embeddings3, embeddings33), axis=2)

embeddings1 = embeddings1.reshape((embeddings1.shape[0], embeddings1.shape[-1]))
embeddings3 = embeddings3.reshape((embeddings3.shape[0], embeddings3.shape[-1]))
print(embeddings1.shape, embeddings3.shape)

if args.classifier == "knn":
    
    
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(embeddings1, y_train)
    y_pred = knn.predict(embeddings3)
    """
    k_values = range(5, 13)
    cv_scores = []

    # Perform 10-fold cross-validation for each k
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, embeddings1, y_train, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())

    # Determine the optimal k
    optimal_k = k_values[np.argmax(cv_scores)]
    print(f"The optimal number of neighbors is {optimal_k}")
    plt.plot(k_values, cv_scores)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()
    """
elif args.classifier == "svm":

    """
    tfidf_vectorizer = TfidfVectorizer()
    # Fit and transform the text data
    X_train = tfidf_vectorizer.fit_transform(train_df["Claim"])
    X_test = tfidf_vectorizer.transform(test_df["Claim"])
    """

    X_train = embeddings1
    X_test = embeddings3
    #sc = StandardScaler()
    #X_train = sc.fit_transform(X_train)
    #X_test = sc.transform(X_test)
    # Create SVM classifier
    svm_classifier = LinearSVC( C=1.0, random_state=42)
    # Train the classifier
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)


print(confusion_matrix(y_test, y_pred))
print(f"""f1-macro: {f1_score(y_test, y_pred, average='macro')}, f1-micro: {f1_score(y_test, y_pred, average='micro')}, f1-weighted: {f1_score(y_test, y_pred, average='weighted')}, prec-score: {precision_score(y_test, y_pred, average='weighted')}, recall-score: {recall_score(y_test, y_pred, average='weighted')}, acc-score: {accuracy_score(y_test, y_pred)}""")
