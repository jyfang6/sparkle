
import torch 
import torch.nn as nn 
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Union, List

from transformers import BertModel, RobertaModel, XLMRobertaForSequenceClassification, T5ForConditionalGeneration
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, MaskedLMOutput

class BertEncoder(BertModel):

    # 基于BertModel的模型都可以用这个encoder，如 
    # bert: bert-base-uncased, 
    # sapbert: cambridgeltl/SapBERT-from-PubMedBERT-fulltext; 
    # pubmedbert: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

    def __init__(self, config, project_dim: int = -1, pool_type: str = "cls", **kwargs):
        
        super().__init__(config)
        assert pool_type in ["cls", "pool", "none", "average"]
        if project_dim > 0:
            self.projection = nn.Linear(config.hidden_size, project_dim) if project_dim > 0 else None
            # NOTE: colbert原本的代码是将向量归一化，而不是使用layernorm 
            self.norm = nn.LayerNorm(project_dim)
        self.embedding_size = project_dim if project_dim > 0 else config.hidden_size 
        self.pool_type = pool_type
        self.kwargs = kwargs 

        self.init_weights()
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
            
        transformer_output = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            return_dict=True
        )
        last_hidden_states = transformer_output.last_hidden_state
        token_embed = last_hidden_states

        if hasattr(self, "projection"):
            token_embed = self.norm(self.projection(token_embed))
            # token_embed = self.projection(token_embed)
            # token_embed = F.normalize(token_embed, p=2, dim=2)

        if self.pool_type == "cls":
            out = (token_embed[:, 0], last_hidden_states)

        elif self.pool_type == "pool" or self.pool_type == "average":
            # out = transformer_output.pooler_output
            extended_attention_mask = attention_mask.unsqueeze(-1)
            masked_token_embed = token_embed * extended_attention_mask
            token_length = extended_attention_mask.sum(1)
            pool_token_embed = masked_token_embed.sum(1) / torch.where(token_length==0, 1, token_length)
            out = (pool_token_embed, last_hidden_states)
        
        elif self.pool_type == "none":
            out = (token_embed, last_hidden_states)

        else:
            raise NotImplemented(f"{self.pool_type} is not implemented!")
        
        return out


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class E5Encoder(BertModel):

    def __init__(self, config, add_pooling_layer=True, **kwargs):
        super().__init__(config, add_pooling_layer)
        self.kwargs = kwargs

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        transformer_output = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            return_dict=True
        )
        last_hidden_states = transformer_output.last_hidden_state
        embeddings = average_pool(last_hidden_states, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    

class ContrieverEncoder(BertModel):

    def __init__(self, config, add_pooling_layer=True, **kwargs):
        super().__init__(config, add_pooling_layer)
        self.kwargs = kwargs

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        transformer_output = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            return_dict=True
        )
        last_hidden_states = transformer_output.last_hidden_state
        last_hidden_states = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embeddings = last_hidden_states.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        return embeddings
    

class BGEEncoder(BertModel):

    def __init__(self, config, add_pooling_layer=True, **kwargs):
        super().__init__(config, add_pooling_layer)
        self.kwargs = kwargs
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):

        transformer_output = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            return_dict=True
        )
        
        last_hidden_states = transformer_output.last_hidden_state
        embeddings = last_hidden_states[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class BGERerankerEncoder(XLMRobertaForSequenceClassification):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.kwargs = kwargs
    
    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        transformer_output = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids, 
            return_dict=True
        )
        logits = transformer_output.logits
        return logits

    
class MonoT5RerankerEncoder(T5ForConditionalGeneration):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.kwargs = kwargs 
    
    def forward( 
        self, 
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs, 
    ):
        transformer_output = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            decoder_input_ids=decoder_input_ids, 
            decoder_attention_mask=decoder_attention_mask,
            labels=labels, 
            return_dict=True
        )

        if labels is not None:
            return (transformer_output.loss, transformer_output.logits)
        else:
            return (transformer_output.logits, )


def load_beamdr_checkpoint(model_name_or_path, **kwargs):
    beamdr =  BeamDREncoder(**kwargs)
    checkpoint = torch.load(model_name_or_path, map_location=beamdr.question_model.device)
    beamdr.load_state_dict(checkpoint["model_dict"], strict=True)
    return beamdr

class BeamDREncoder(nn.Module):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.question_model = BertModel.from_pretrained("bert-base-uncased")
        self.ctx_model = BertModel.from_pretrained("bert-base-uncased")
        self.config = self.question_model.config
    
    def query_embed(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        transformer_output = self.question_model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            return_dict=True
        )
        embeddings = transformer_output.last_hidden_state[:, 0]
        return embeddings
    
    def doc_embed(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        transformer_output = self.ctx_model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            return_dict=True
        )
        embeddings = transformer_output.last_hidden_state[:, 0]
        return embeddings


class OpenAITextEmbeddingEncoder(nn.Module):

    def __init__(self, model_name_or_path: str, **kwargs) -> None:
        super().__init__()
        from openai import OpenAI
        from setup.setup import OPENROUTER_API_KEY 
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )
        self.model_name = model_name_or_path
        self.embedding_size = 1536  # Example size for text-embedding-ada-002
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
            encoding_format="float"
        )
        embeddings = [item.embedding for item in response.data]
        return torch.tensor(embeddings)

    def query_embed(self, query_texts: List[str]) -> torch.Tensor:
        return self.embed(query_texts)
    
    def doc_embed(self, doc_texts: List[str]) -> torch.Tensor:
        return self.embed(doc_texts) 
    