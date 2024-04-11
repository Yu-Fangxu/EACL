import torch
import torch.nn as nn
from transformers import AutoModel
from sklearn.metrics import euclidean_distances
import torch.nn.functional as F
def extract_bert_embeddings(tokenizer, model, text, device):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Forward pass through BERT model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract last layer hidden states
    last_hidden_state = outputs.last_hidden_state

    # Average pooling across tokens to get sentence embeddings
    sentence_embedding = torch.mean(last_hidden_state, dim=1)
    
    return sentence_embedding

class CLModel(nn.Module):
    def __init__(self, args, n_classes, tokenizer=None):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_classes = n_classes
        self.pad_value = args.pad_value
        self.mask_value = 50265
        self.f_context_encoder = AutoModel.from_pretrained(args.bert_path)
        
        num_embeddings, self.dim = self.f_context_encoder.embeddings.word_embeddings.weight.data.shape
        self.avg_dist = []

        self.f_context_encoder.resize_token_embeddings(num_embeddings + 256)
        self.eps = 1e-8
        self.device = "cuda" if self.args.cuda else "cpu"
        self.predictor = nn.Sequential(
            # nn.Linear(self.dim, self.dim),
            # nn.ReLU(),
            nn.Linear(self.dim, self.num_classes)
        )
        self.map_function = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, args.mapping_lower_dim),
        ).to(self.device)

        self.tokenizer = tokenizer

        if args.dataset_name == "IEMOCAP":
            self.emo_proto = torch.load(f"{args.proto_path}/iemocap_emo.pt").to(self.device)
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5]).to(self.device)
        elif args.dataset_name == "MELD":
            self.emo_proto = torch.load(f"{args.proto_path}/meld_emo.pt").to(self.device)
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        elif args.dataset_name == "EmoryNLP":
            self.emo_proto = torch.load(f"{args.proto_path}/emorynlp_emo.pt").to(self.device)
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6]).to(self.device)

    def device(self):
        return self.f_context_encoder.device
    
    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1))/2 + self.eps
    
    def _forward(self, sentences):
        mask = 1 - (sentences == (self.pad_value)).long()

        utterance_encoded = self.f_context_encoder(
            input_ids=sentences,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']
        mask_pos = (sentences == (self.mask_value)).long().max(1)[1]
        mask_outputs = utterance_encoded[torch.arange(mask_pos.shape[0]), mask_pos]
        mask_mapped_outputs = self.map_function(mask_outputs)
        feature = torch.dropout(mask_outputs, self.dropout, train=self.training)
        feature = self.predictor(feature)
        if self.args.use_nearest_neighbour:
            protos = self.map_function(self.emo_proto)

            self.last_emo_proto = protos
            proto_scores = self.score_func(mask_mapped_outputs.unsqueeze(1), protos.unsqueeze(0))
            
        else:
            proto_scores = None
        return feature, mask_mapped_outputs, mask_outputs, proto_scores
    
    def forward(self, sentences, return_mask_output=False):
        '''
        generate vector representations for each turn of conversation
        '''
        feature, mask_mapped_outputs, mask_outputs, proto_scores = self._forward(sentences)
        
        if return_mask_output:
            return feature, mask_mapped_outputs, mask_outputs, proto_scores
        else:
            return feature
        
class Classifier(nn.Module):
    def __init__(self, args, protos) -> None:
        super(Classifier, self).__init__()
        self.weight = nn.Parameter(protos)
        self.args = args
    
    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1))/2 + 1e-8
    
    def forward(self, emb):
        return self.score_func(self.weight.unsqueeze(0), emb.unsqueeze(1)) / self.args.temp
