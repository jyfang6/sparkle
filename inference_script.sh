
dataset=2wikimultihopqa 
index_folder=/path/to/index/folder
index_embedding_size=1024
cached_kg_triples_file=/path/to/cached/kg_corpus_folder
policy_model_name_or_path=/path/to/proxy/model/checkpoint
reasoning_model_name_or_path=qwen2.5_7b_instruct
save_dir=/path/to/checkpoint/folder
name=inference_${dataset}

python -m main \
    --dataset_yaml hparams/dataset/end_to_end_retrieval/${dataset}_retrieval_e5.yaml \
    --retriever_yaml hparams/retriever/e5_retriever.yaml \
    --index_folder ${index_folder} \
    --embedding_size ${index_embedding_size} \
    --kg_generator_yaml hparams/knowledge_graph/kg_generator.yaml \
    --cached_kg_triples_file ${cached_kg_triples_file} \
    --kg_adaptive_rag_yaml hparams/model/kg_adaptive_rag.yaml \
    --policy_model_name_or_path ${policy_model_name_or_path} \
    --reasoning_model_name_or_path ${reasoning_model_name_or_path} \
    --query_file /path/to/test_data \
    --sample_k 500 \
    --topk 10 \
    --save_dir ${save_dir} \
    --name ${name} \
    --n_rounds 1 \
    --save_frequency 10 \
    --remove_demo 

done
