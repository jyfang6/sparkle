
import torch
import numpy as np 
import torch.nn as nn 
import torch.distributed as dist
from typing import Optional 

from retriever.encoders import * 
from utils.utils import * 
from data.collators import * 
from data.corpus import * 
from retriever.index import * 

RETRIEVER_MAP = {
    "BertRetriever": BertEncoder,
    "E5Retriever": E5Encoder,
    "ContrieverRetriever": ContrieverEncoder, 
    "BGERetriever": BGEEncoder, 
    # "RobertaRetriever": RobertaEncoder, 
    # "LukeRetriever": LukeEncoder, 
    # "Contriever": Contriever, 
    "BeamDR": BeamDREncoder,
    "OpenAITextEmbeddingRetriever": OpenAITextEmbeddingEncoder, 
}

def load_retriever(retriever_name, model_name_or_path, **kwargs):
    # 支持两种方法，如果model_name_or_path是一个folder的话，直接用.from_pretrained()的方法加载，否则的话用.load_state_dict()加载
    if retriever_name not in RETRIEVER_MAP:
        raise KeyError(f"{retriever_name} is not implemented! Current available retrievers: {list(RETRIEVER_MAP.keys())}")
    print(f"loading {retriever_name} model from {model_name_or_path} ...")
    if retriever_name == "BeamDR":
        return load_beamdr_checkpoint(model_name_or_path, **kwargs)
    if retriever_name == "OpenAITextEmbeddingRetriever":
        return RETRIEVER_MAP[retriever_name](model_name_or_path, **kwargs)
    return RETRIEVER_MAP[retriever_name].from_pretrained(model_name_or_path, **kwargs)


class BaseRetriever(nn.Module):

    def __init__(self, retriever_name, model_name_or_path, retriever_kwargs={}, 
                 temperature=1.0, norm_query=False, norm_doc=False, local_rank=-1, **kwargs):
        
        super().__init__()
        self.encoder = load_retriever(retriever_name, model_name_or_path, **retriever_kwargs, **kwargs)
        self.retriever_name = retriever_name
        self.model_name_or_path = model_name_or_path
        self.retriever_kwargs = retriever_kwargs
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.local_rank = local_rank
        self.world_size = dist.get_world_size() if self.local_rank >= 0 else 1 
        self.temperature = temperature # TODO: 可以研究一下temperate取值的影响，我记得atalas中是设置为hidden_size的开根号，contriever中设为0.05，dpr中为1.0 
        self.kwargs = kwargs

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine == True:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    @property
    def device(self):
        for n, p in self.named_parameters():
            return p.device 
    
    @property
    def hidden_size(self):
        return self.encoder.config.hidden_size
    
    # def query(self, *args, **kwargs):
    #     raise NotImplementedError("query function is not implemented!")

    # def doc(self, *args, **kwargs):
    #     raise NotImplementedError("doc function is not implemented!")
    
    def compute_logits(self, query_embeddings, doc_embeddings, **kwargs):
        # 默认使用内积来计算query和document相似度
        if len(query_embeddings.shape) ==  1 and len(doc_embeddings.shape) == 1:
            logits = torch.einsum("d,d->", query_embeddings, doc_embeddings)
        elif len(query_embeddings.shape) ==  1 and len(doc_embeddings.shape) == 2:
            # 单个query和多个document的情况
            logits = torch.einsum("d,md->m", query_embeddings, doc_embeddings)
        elif len(query_embeddings.shape) == 2 and len(doc_embeddings.shape) == 3:
            # 多个query和多个document: 计算一个query和它对应的那部分document的相似度
            assert len(query_embeddings) == len(doc_embeddings)
            # logits = torch.sum(query_embeddings.unsqueeze(1)*doc_embeddings, dim=-1) 
            logits = torch.einsum("nd,nmd->nm", query_embeddings, doc_embeddings)
        elif len(query_embeddings.shape) == 2 and len(doc_embeddings.shape) == 2:
            # 多个query和多个document: 计算每一个query和所有document的相似度
            logits = torch.einsum("nd,md->nm", query_embeddings, doc_embeddings) # n x m 
        else:
            raise ValueError(f"Invalid embedding shape! query_embeddings: {query_embeddings.shape}, doc_embeddings: {doc_embeddings.shape}.")

        return logits

    def score(self, query_embeddings, doc_embeddings, **kwargs):
        # 默认使用内积（即计算每一个query和所有passages的内积）以及temperature(可以设置为embedding的维度开根号)计算相似度
        # temperature = kwargs.get("temperature", 1.0)
        # scores = self.compute_logits(query_embeddings, doc_embeddings) / temperature
        if self.temperature == "sqrt":
            scores = self.compute_logits(query_embeddings, doc_embeddings) / np.sqrt(query_embeddings.shape[-1])
        else:
            scores = self.compute_logits(query_embeddings, doc_embeddings) / self.temperature
        return scores
    
    def get_encoder_output(self, args, **kwargs):
        # 这个函数就假设了输入的input_ids都是二维的了，输出就是用于计算相似度的embedding，后面继承的时候主要也是覆盖这个函数，
        assert len(args["input_ids"].shape) == 2 
        outputs = self.encoder(**args, **kwargs)
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs 

    def encoder_embed(self, args, **kwargs):

        need_reshape = (len(args["input_ids"].shape) != 2)
        if need_reshape:
            *other_dim, last_dim = args["input_ids"].shape 
            args = {k: v.reshape(-1, last_dim) if torch.is_tensor(v) else v for k, v in args.items()}
        embeddings = self.get_encoder_output(args, **kwargs)
        embedding_size = embeddings.shape[-1]
        if need_reshape:
            embeddings = embeddings.reshape(*other_dim, embedding_size) # 这里假设了对于每一个document都只有一个embedding，所以不用考虑last_dim 
        return embeddings 
    
    def query(self, args, **kwargs):
        query_embeddings = self.encoder_embed(args, **kwargs)
        if self.norm_query:
            query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=-1)
        return query_embeddings
    
    def doc(self, args, **kwargs):
        doc_embeddings = self.encoder_embed(args, **kwargs)
        if self.norm_doc:
            doc_embeddings = torch.nn.functional.normalize(doc_embeddings, dim=-1)
        return doc_embeddings
    
    def save_model(self, save_path):
        self.encoder.save_pretrained(save_path)
    
    def load_model(self, save_path):
        self.encoder = load_retriever(self.retriever_name, save_path, **self.retriever_kwargs, **self.kwargs)
        

class InBatchRetriever(BaseRetriever):

    def forward(self, query_args, doc_args, labels=None, **kwargs):
        
        """
        要求每一个query有相同的positive documents和negative documents, 否则会多卡跑的话会报错
        """
        
        query_embeddings = self.query(query_args, **kwargs)
        global_query_embeddings = get_global_embeddings_for_inbatchtraining(self.local_rank, self.world_size, query_embeddings)
        doc_embeddings = self.doc(doc_args, **kwargs)
        global_doc_embeddings = get_global_embeddings_for_inbatchtraining(self.local_rank, self.world_size, doc_embeddings)
        local_doc_size = len(doc_embeddings)
        global_labels = get_global_labels_for_inbatchtraining(self.local_rank, self.world_size, labels, local_doc_size)
        # scores = self.score(global_query_embeddings, global_doc_embeddings, temperature=self.temperature)
        scores = self.score(global_query_embeddings, global_doc_embeddings)

        if global_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(scores, global_labels)
            outputs = (loss, scores, global_query_embeddings, global_doc_embeddings)
        else:
            outputs = (scores, global_query_embeddings, global_doc_embeddings)
        
        return outputs 


class APIRetriever(BaseRetriever):
    
    def encoder_embed(self, args, **kwargs):
        # args is expected to a str or List[str] 
        embeddings = self.encoder.forward(args)
        return embeddings 


class MoCoRetriever(BaseRetriever):

    def __init__(self, retriever_name, model_name_or_path, **kwargs):
        self.encoder = load_retriever(retriever_name, model_name_or_path, **kwargs)

    def save_model(self):
        pass


class DualRetriever(InBatchRetriever):

    def encoder_embed(self, embed_function, args, **kwargs):

        need_reshape = (len(args["input_ids"].shape) != 2)
        if need_reshape:
            *other_dim, last_dim = args["input_ids"].shape 
            args = {k: v.reshape(-1, last_dim) if torch.is_tensor(v) else v for k, v in args.items()}

        outputs = embed_function(**args, **kwargs) # encoder可能返回多个值，但是只取第一个变量作为embedding
        if isinstance(outputs, (tuple, list)):
            embeddings = outputs[0]
        else:
            embeddings = outputs

        embedding_size = embeddings.shape[-1]
        if need_reshape:
            embeddings = embeddings.reshape(*other_dim, embedding_size) # 这里假设了对于每一个document都只有一个embedding，所以不用考虑last_dim 
        return embeddings 

    def query(self, args, **kwargs):
        query_embeddings = self.encoder_embed(self.encoder.query_embed, args, **kwargs)
        if self.norm_query:
            query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=-1)
        return query_embeddings 
    
    def doc(self, args, **kwargs):
        doc_embeddings = self.encoder_embed(self.encoder.doc_embed, args, **kwargs)
        if self.norm_doc:
            doc_embeddings = torch.nn.functional.normalize(doc_embeddings, dim=-1)
        return doc_embeddings
    
    # TODO: 目前计算loss使用InBatch Negatives的方法，但是缺少了保存模型的代码
     

class RetrieverWithReaderScores(BaseRetriever):

    def forward(self, query_args, doc_args, gold_scores=None, **kwargs):

        query_embeddings = self.query(query_args, **kwargs)
        doc_embeddings = self.doc(doc_args, **kwargs)
        # scores = self.score(query_embeddings, doc_embeddings, temperature=self.temperature)
        # 参考FiD训练retriever模型的方法，把内积除以embedding_size的开根号
        # scores = self.score(query_embeddings, doc_embeddings, temperature=np.sqrt(doc_embeddings.shape[-1]))
        scores = self.score(query_embeddings, doc_embeddings)
        
        if gold_scores is not None:
            # TODO: Atalas 中还有 emdr loss，它是基于logprob 函数计算的
            loss_fct = nn.KLDivLoss()
            logits = torch.nn.functional.log_softmax(scores.float(), dim=-1)
            target = gold_scores if is_normalized_last_dim(gold_scores) else nn.functional.softmax(gold_scores, dim=-1)
            loss = loss_fct(logits, target.float())
            outputs = (loss, scores, query_embeddings, doc_embeddings)
        else:
            outputs = (scores, query_embeddings, doc_embeddings)
        
        return outputs 


class KGChainRetrieverWithReaderScores(BaseRetriever):

    def forward(self, query_args, doc_args, labels=None, scores=None, **kwargs):
        
        """
        要求每一个query有相同的positive documents和negative documents, 否则会多卡跑的话会报错(主要是在得到global labels和计算基于scores的loss的时候会报错)
        """
        
        query_embeddings = self.query(query_args, **kwargs)
        global_query_embeddings = get_global_embeddings_for_inbatchtraining(self.local_rank, self.world_size, query_embeddings)
        local_query_size = len(query_embeddings)
        doc_embeddings = self.doc(doc_args, **kwargs)
        global_doc_embeddings = get_global_embeddings_for_inbatchtraining(self.local_rank, self.world_size, doc_embeddings)
        local_doc_size = len(doc_embeddings)
        global_labels = get_global_labels_for_inbatchtraining(self.local_rank, self.world_size, labels, local_doc_size)
        # scores = self.score(global_query_embeddings, global_doc_embeddings, temperature=self.temperature)
        retriever_scores = self.score(global_query_embeddings, global_doc_embeddings)
        
        if global_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(retriever_scores, global_labels)
            if scores is not None:
                lamb = self.kwargs.get("lambda", 1.0)
                local_retriever_scores = self.score(query_embeddings, doc_embeddings.reshape(local_query_size, -1, doc_embeddings.shape[-1]))
                local_retriever_probs = torch.softmax(local_retriever_scores, dim=1)
                local_reader_probs = torch.softmax(torch.log(scores.reshape(local_query_size, -1)+1e-6), dim=1)
                loss += -1 * lamb * torch.mean(torch.logsumexp(torch.log(local_retriever_probs+1e-6)+torch.log(local_reader_probs+1e-6), dim=1))
            """
            if scores is not None:
                lamb = self.kwargs.get("lambda", 1.0)
                num_queries = len(query_embeddings)
                ip_retriever_scores = torch.matmul(query_embeddings, doc_embeddings.T)
                for i in range(num_queries):
                    start_idx = labels[i]
                    end_idx = labels[i+1] if i+1 < num_queries else len(doc_embeddings)
                    reader_scores = torch.softmax(torch.log(scores[start_idx: end_idx]+1e-6), dim=0)
                    rs = torch.softmax(ip_retriever_scores[i, start_idx: end_idx]/self.temperature, dim=0)
                    loss += -1 * lamb * torch.logsumexp(torch.log(reader_scores.float()+1e-6)+torch.log(rs.float()+1e-6), dim=0) / num_queries
            """
            outputs = (loss, scores, global_query_embeddings, global_doc_embeddings)
        else:
            outputs = (scores, global_query_embeddings, global_doc_embeddings)
        return outputs 


class DenseRetriever(nn.Module):

    def __init__(
        self, 
        retriever: BaseRetriever, 
        collator: RetrieverCollator, 
        indexer: Optional[Indexer]=None, 
        corpus: Optional[Corpus]=None, 
        batch_size: int=4,
        **kwargs 
    ):
        super().__init__()
        self.retriever = retriever
        self.device=self.retriever.device
        self.retriever.eval()

        self.collator = collator
        self.indexer = indexer
        self.corpus = corpus

        self.batch_size = batch_size
        self.kwargs = kwargs
    
    def get_documents(self, docid_list: Union[List[str], Dict[str, float]]) -> List[dict]:

        documents = []
        if isinstance(docid_list, list):
            for docid in docid_list:
                documents.append(deepcopy(self.corpus.get_document(docid)))
        elif isinstance(docid_list, dict):
            # rank documents based on scores 
            sorted_docid_list = sorted(docid_list.items(), key=lambda x: x[1], reverse=True)
            for docid, score in sorted_docid_list:
                document = deepcopy(self.corpus.get_document(docid))
                document["score"] = float(score)
                documents.append(document)
        else:
            raise ValueError(f"{type(docid_list)} is not a supported type for \"docid_list\"!")

        return documents
    
    def calculate_query_embeddings(self, queries: List[str], max_length: int=None, verbose: bool=False, **kwargs) -> Tensor:

        queries_embeddings_list = [] 
        assert isinstance(queries, list) and len(queries) > 0 # must provide queries 
        num_batches = (len(queries) - 1) // self.batch_size + 1 
        if verbose:
            progress_bar = trange(num_batches, desc="Calculating Query Embeddings")
        for i in range(num_batches):
            batch_queries = queries[i*self.batch_size: (i+1)*self.batch_size]
            batch_queries_inputs = self.collator.encode_query(batch_queries, max_length=max_length, **kwargs)
            batch_queries_inputs = to_device(batch_queries_inputs, self.device)
            batch_queries_embeddings = self.retriever.query(batch_queries_inputs).detach().cpu()
            queries_embeddings_list.append(batch_queries_embeddings)
            if verbose:
                progress_bar.update(1)
        # queries_embeddings = np.concatenate(queries_embeddings_list, axis=0)
        queries_embeddings = torch.cat(queries_embeddings_list, dim=0)

        return queries_embeddings 
    
    def calculate_document_embeddings(self, documents: List[str], max_length: int=None, verbose: bool=False, **kwargs) -> Tensor:

        documents_embeddings_list = [] 
        assert isinstance(documents, list) and len(documents) > 0 # must provide documents
        num_batches = (len(documents) - 1) // self.batch_size + 1 
        if verbose:
            progress_bar = trange(num_batches, desc="Calculating Document Embeddings")
        for i in range(num_batches):
            batch_documents = documents[i*self.batch_size: (i+1)*self.batch_size]
            batch_documents_inputs = self.collator.encode_doc(batch_documents, max_length=max_length, **kwargs)
            batch_documents_inputs = to_device(batch_documents_inputs, self.device)
            batch_documents_embeddings = self.retriever.doc(batch_documents_inputs).detach().cpu()
            documents_embeddings_list.append(batch_documents_embeddings)
            if verbose:
                progress_bar.update(1)
        # documents_embeddings = np.concatenate(documents_embeddings_list, axis=0)
        documents_embeddings = torch.cat(documents_embeddings_list, dim=0)

        return documents_embeddings 

    def parse_indexer_output(self, indexer_output):

        retrieval_results = []
        for topk_str_indices, topk_score_array in indexer_output:
            one_retrieval_results = [] 
            for docid, score in zip(topk_str_indices, topk_score_array):
                if self.corpus is not None:
                    document = deepcopy(self.corpus.get_document(docid))
                    document["score"] = float(score)
                else:
                    document = {"id": docid, "score": score}
                one_retrieval_results.append(document)
            retrieval_results.append(one_retrieval_results)
        
        return retrieval_results

    def batch_retrieve(self, queries: List[str], topk: int, verbose: bool=False, **kwargs) -> List[dict]:

        # queries_embeddings_list = [] 
        # num_batches = (len(queries) - 1) // self.batch_size + 1 
        # if verbose:
        #     progress_bar = trange(num_batches, desc="Calculating Query Embedding")
        # for i in range(num_batches):
        #     batch_queries = queries[i*self.batch_size: (i+1)*self.batch_size]
        #     batch_queries_inputs = self.collator.encode_query(batch_queries)
        #     batch_queries_inputs = to_device(batch_queries_inputs, self.device)
        #     batch_queries_embeddings = self.retriever.query(batch_queries_inputs).detach().cpu().numpy()
        #     queries_embeddings_list.append(batch_queries_embeddings)
        #     if verbose:
        #         progress_bar.update(1)
        # queries_embeddings = np.concatenate(queries_embeddings_list, axis=0)
        # obtain query embeddings 
        queries_embeddings = self.calculate_query_embeddings(queries=queries, verbose=verbose, **kwargs)
        queries_embeddings = queries_embeddings.numpy()

        # knn search 
        knn_results = self.indexer.search_knn(
            query_vectors=queries_embeddings, 
            top_docs=topk, 
            index_batch_size=1024,
            verbose=verbose
        )

        # parse results 
        retrieval_results = []
        for topk_str_indices, topk_score_array in knn_results:
            one_retrieval_results = [] 
            for docid, score in zip(topk_str_indices, topk_score_array):
                if self.corpus is not None:
                    document = deepcopy(self.corpus.get_document(docid))
                    document["score"] = float(score)
                else:
                    document = {"id": docid, "score": score}
                one_retrieval_results.append(document)
            retrieval_results.append(one_retrieval_results)
        return retrieval_results
    
    def forward(self, queries: Union[str, List[str]], topk: int, verbose: bool=False, **kwargs) -> Union[dict, List[dict]]:
        """
        Input:
            queries: str or List[str]
        Output:
            if self.corpus is not None:
                [ [{doc_content_in_corpus(key-value pair), "score": float}, ...topk], ... ]
            else:
                [ [{"id": docid[str], "score": doc_score[float]}, ...], ... ]
        """
        assert self.indexer is not None # must provide indexer 
        if isinstance(queries, str):
            return self.batch_retrieve([queries], topk=topk, verbose=verbose, **kwargs)[0]
        else:
            return self.batch_retrieve(queries, topk=topk, verbose=verbose, **kwargs)


class VectorPRF(nn.Module):

    def __init__(
        self, 
        retriever: DenseRetriever,
        num_pseudo_relevant_docs: int=3,
        prf_type: str = "rocchio", 
        alpha: float = 0.6, 
        beta: float = 0.4, 
        **kwargs, 
    ):
        super().__init__()
        assert prf_type in ["average", "rocchio"] # "prf_type" must be chosen from ["average", "rocchio"]
        self.retriever = retriever 
        self.num_pseudo_relevant_docs = num_pseudo_relevant_docs
        self.prf_type = prf_type
        self.alpha = alpha
        self.beta = beta
        self.kwargs = kwargs

    def calculate_prf_query_embeddings(self, query_embeddings: torch.Tensor, pseudo_relevant_document_embeddings: torch.Tensor):

        if self.prf_type == "average":
            query_doc_embeddings = torch.cat([query_embeddings.unsqueeze(1), pseudo_relevant_document_embeddings], dim=1)
            updated_query_embeddings = query_doc_embeddings.mean(dim=1)
        elif self.prf_type == "rocchio":
            updated_query_embeddings = self.alpha * query_embeddings + self.beta * pseudo_relevant_document_embeddings.mean(dim=1)
        else:
            raise ValueError(f"{self.prf_type} is an invalid prf_type!")
        
        return updated_query_embeddings

    def batch_retrieve(self, queries: List[str], topk: int, verbose: bool=False, **kwargs) -> List[dict]:
        
        # initial retrieval 
        num_queries = len(queries)
        init_query_embeddings = self.retriever.calculate_query_embeddings(queries=queries)
        init_knn_results = self.retriever.indexer.search_knn(
            query_vectors = init_query_embeddings.numpy(),
            top_docs = self.num_pseudo_relevant_docs,
            verbose = verbose
        )
        
        all_pseudo_relevant_documents = [] 
        for topk_docids, topk_scores in init_knn_results:
            for docid in topk_docids:
                document = self.retriever.corpus.get_document_text(docid)
                all_pseudo_relevant_documents.append(document)
        
        # calculate document embeddings 
        pseudo_relevant_document_embeddings = self.retriever.calculate_document_embeddings(all_pseudo_relevant_documents)
        pseudo_relevant_document_embeddings = pseudo_relevant_document_embeddings.reshape(num_queries, self.num_pseudo_relevant_docs, -1)

        # calculate updated query embeddings
        query_embeddings = self.calculate_prf_query_embeddings(
            query_embeddings=init_query_embeddings, 
            pseudo_relevant_document_embeddings=pseudo_relevant_document_embeddings
        )
        knn_results = self.retriever.indexer.search_knn(
            query_vectors=query_embeddings.numpy(), 
            top_docs=topk, 
            index_batch_size=1024, 
            verbose=verbose
        )
        retrieval_results = self.retriever.parse_indexer_output(knn_results)

        return retrieval_results
    
    def forward(self, queries: Union[str, List[str]], topk: int, verbose: bool=False, **kwargs) -> Union[dict, List[dict]]:
        if isinstance(queries, str):
            return self.batch_retrieve([queries], topk=topk, verbose=verbose, **kwargs)[0]
        else:
            return self.batch_retrieve(queries, topk=topk, verbose=verbose, **kwargs)

    

