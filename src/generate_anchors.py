import os
import torch
from transformers import AutoModel, AutoTokenizer, pipeline
import argparse
import warnings

from utils.data_process import *
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "1"

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str, default='princeton-nlp/sup-simcse-roberta-large')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_parser()

    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)

    model = AutoModel.from_pretrained(args.bert_path)
    model.eval()
    save_path = args.bert_path.split("/")[-1]
    iemocap_emos = [
        "neutral",
        "excited",
        "frustrated",
        "sad",
        "happy",
        "angry"
    ]

    meld_emos = [
        'anger',
        'disgust',
        'fear',
        'joy',
        'sadness',
        'surprise',
        'neutral'
    ]

    emorynlp_emos = [
        'joyful',
        'neutral',
        'powerful',
        'mad',
        'scared',
        'peaceful',
        'sad'
    ]
    embeddings = []
    feature_extractor = pipeline("feature-extraction",framework="pt",model=args.bert_path)
    embeddings = []
    with torch.no_grad():
        for emo in iemocap_emos:
            emb = torch.tensor(feature_extractor(emo,return_tensors = "pt")[0]).mean(0)
            embeddings.append(emb.unsqueeze(0))
        embeddings = torch.cat(embeddings, dim=0)
        torch.save(embeddings, f"./emo_anchors/{save_path}/iemocap_emo.pt")

    embeddings = []
    with torch.no_grad():
        for emo in meld_emos:
            emb = torch.tensor(feature_extractor(emo,return_tensors = "pt")[0]).mean(0)
            embeddings.append(emb.unsqueeze(0))
        embeddings = torch.cat(embeddings, dim=0)
        torch.save(embeddings, f"./emo_anchors/{save_path}/meld_emo.pt")

    embeddings = []
    with torch.no_grad():
        for emo in emorynlp_emos:
            emb = torch.tensor(feature_extractor(emo,return_tensors = "pt")[0]).mean(0)
            embeddings.append(emb.unsqueeze(0))
        embeddings = torch.cat(embeddings, dim=0)
        torch.save(embeddings, f"./emo_anchors/{save_path}/emorynlp_emo.pt")