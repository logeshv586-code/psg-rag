import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

class LocalSentenceTransformer:
    def __init__(self, model_name_or_path, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading {model_name_or_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

    def encode(self, sentences, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        if isinstance(sentences, str):
            sentences = [sentences]
            
        all_embeddings = []
        
        # Batch processing
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(self.device)
            
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
            # Mean Pooling - Take attention mask into account for correct averaging
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            if convert_to_numpy:
                all_embeddings.append(sentence_embeddings.cpu().numpy())
            else:
                all_embeddings.append(sentence_embeddings)
                
        if convert_to_numpy:
            return np.concatenate(all_embeddings, axis=0)
        else:
            return torch.cat(all_embeddings, dim=0)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
