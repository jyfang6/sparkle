import os
import glob
import pickle
import logging
import argparse
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import re
logging.getLogger("httpx").setLevel(logging.ERROR)

from transformers import AutoTokenizer
from retriever.retrievers import DenseRetriever
from knowledge_graph.kg_generator import KGGenerator, APIKGGenerator
from model.adaptive_rag import KGAdaptiveRAG
from baselines.search_o1 import extract_answer

from utils.const import *
from utils.pipeline_utils import load_llm_tokenizer_and_model

import sys
from setup.setup import *
sys.path.append(COMMON_FOLDER)
from my_utils import *
from my_evaluation import *

logger = logging.getLogger(__file__)

device = torch.device("cuda")


def setup_parser():

    def my_bool(s):
        if s is None or s.lower() == "none":
            return None
        return bool(s)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)


    parser.add_argument("--dataset_yaml", required=True, type=str, default=None, help="用来得到检索模型的collator")
    parser.add_argument("--retriever_yaml", required=True, type=str, default=None, help="得到检索模型的配置")
    parser.add_argument("--retriever_model_name_or_path", type=str, default=None, help="得到检索模型的路径")


    parser.add_argument("--index_name", type=str, default="ip_indexer")
    parser.add_argument("--index_folder", type=str, required=True, help="index的文件夹")
    parser.add_argument("--embedding_size", type=int, required=True, help="index的embedding size")


    parser.add_argument("--kg_generator_yaml", required=True, type=str, default=None)
    parser.add_argument("--cached_kg_triples_file", type=str, default=None)


    parser.add_argument("--kg_adaptive_rag_yaml", required=True, type=str, default=None, help="kg adaptive rag的配置")
    parser.add_argument("--policy_model_name_or_path", type=str, default=None, help="策略模型的名称或路径")
    parser.add_argument("--reasoning_model_name_or_path", type=str, default=None, help="推理模型的名称或路径")
    parser.add_argument("--remove_demo", action="store_true", help="是否从策略模型中移除demonstration")


    parser.add_argument("--query_file", required=True, type=str, help="query file")
    parser.add_argument("--sample_k", type=int, default=-1, help="sample k examples from the query file")


    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--per_gpu_batch_size", type=int, default=4, help="batch size for retriever model")


    parser.add_argument("--save_dir", type=str, default="checkpoint")
    parser.add_argument("--name", type=str, default="debug")
    parser.add_argument("--n_rounds", type=int, default=1)
    parser.add_argument("--save_intermediate_results", action="store_true")
    parser.add_argument("--save_frequency", type=int, default=None)


    parser.add_argument("--split_data", action="store_true")
    parser.add_argument("--split_batch_size", type=int, default=100)
    parser.add_argument("--num_splits", type=int, default=None)
    parser.add_argument("--split_index", type=int, default=0)


    parser.add_argument("--ablation1", action="store_true", help="是否使用ablation1的策略")
    parser.add_argument("--use_thought", action="store_true", help="是否在策略模型生成结果的时候使用thought")

    args = parser.parse_args()
    args.dataset = parse_yaml(args.dataset_yaml)
    args.retriever = parse_yaml(args.retriever_yaml)
    args.kg_generator = parse_yaml(args.kg_generator_yaml)
    args.kg_adaptive_rag = parse_yaml(args.kg_adaptive_rag_yaml)

    if args.retriever_model_name_or_path is not None:
        print(f"args.retriever_model_name_or_path is not None, setting args.retriever.model_name_or_path to {args.retriever_model_name_or_path}.")
        args.retriever.model_name_or_path = args.retriever_model_name_or_path
    else:
        print(f"`retriever_model_name_or_path` is not set, will use the `model_name_or_path` {args.retriever.model_name_or_path} in the retriever_yaml!")

    if args.policy_model_name_or_path is not None:
        print(f"args.policy_model_name_or_path is not None, setting args.kg_adaptive_rag.policy_model_config.model_name_or_path to {args.policy_model_name_or_path}.")
        args.kg_adaptive_rag.policy_model_config.model_name_or_path = args.policy_model_name_or_path
    else:
        print(f"`policy_model_name_or_path` is not set, will use the `model_name_or_path` {args.kg_adaptive_rag.policy_model_config.model_name_or_path} in the kg_adaptive_rag_yaml!")

    if args.remove_demo:
        print("args.remove_demo is True, Removing demonstration from policy model configuration.")
        args.kg_adaptive_rag.policy_model_config.use_demo = False
    
    if args.reasoning_model_name_or_path is not None:
        print(f"args.reasoning_model_name_or_path is not None, setting args.kg_adaptive_rag.reasoning_model_config.model_name_or_path to {args.reasoning_model_name_or_path}.")
        args.kg_adaptive_rag.reasoning_model_config.model_name_or_path = args.reasoning_model_name_or_path
    else:
        print(f"`reasoning_model_name_or_path` is not set, will use the `model_name_or_path` {args.kg_adaptive_rag.reasoning_model_config.model_name_or_path} in the kg_adaptive_rag_yaml!")

    if args.topk is not None:
        print(f"args.topk is not None, setting args.kg_adaptive_rag.topk to {args.topk}.")
        args.kg_adaptive_rag.topk = args.topk
    else:
        print(f"`topk` is not set, will use the `topk` {args.kg_adaptive_rag.topk} in the kg_adaptive_rag_yaml!")

    return args


def load_existing_results(save_prediction_file):

    with open(save_prediction_file, "rb") as f:
        predictions = pickle.load(f)

    acc_list, em_list, f1_list = [], [], []
    for prediction in predictions:
        acc_list.append(prediction["acc"])
        em_list.append(prediction["em"])
        f1_list.append(prediction["f1"])

    return {"acc": np.mean(acc_list), "em": np.mean(em_list), "f1": np.mean(f1_list)}


def is_multihop_qa(args):
    if args.dataset.name.lower() in ["nq", "tqa", "asqa"]:
        return False
    elif args.dataset.name.lower() in ["hotpotqa", "2wikimultihopqa", "musique", "webqa", "bamboogle", "hotpotqa_hipporag", "2wikimultihopqa_hipporag", "musique_hipporag"]:
        return True
    else:
        raise ValueError(f"Unknown dataset: {args.dataset.name}")

def is_long_form_qa(args):
    if args.dataset.name.lower() in ["asqa"]:
        return True
    return False

def extract_non_reasoning_model_answer(output: str) -> str:

    original_output = output


    pattern1 = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern1, output, re.DOTALL)
    for match in matches:
        answer = match.replace("<answer>", "").replace("</answer>", "").strip()
        if answer:
            return answer


    ANSWER_TAG = "ANSWER_TAG"
    output = output.replace("<answer>", ANSWER_TAG).replace("</answer>", ANSWER_TAG)
    segments = re.split(ANSWER_TAG, output)
    for segment in segments:
        segment = segment.strip()
        if segment:
            return segment


    print(f"Warning: Failed to extract answer from output: {original_output}")
    return original_output

def qa_evaluate(args, questions: List[Dict], model: KGAdaptiveRAG, round: int):

    multihop_qa = is_multihop_qa(args)
    long_form_qa = is_long_form_qa(args)

    acc_list, em_list, f1_list = [], [], []
    rouge_list = []


    existing_predictions = []
    questions_that_have_predictions = dict()
    if os.path.exists(os.path.join(args.save_dir, args.name, f"round_{round}_predictions.pkl")):
        with open(os.path.join(args.save_dir, args.name, f"round_{round}_predictions.pkl"), "rb") as f:
            existing_predictions = pickle.load(f)
        for prediction in existing_predictions:

            questions_that_have_predictions[prediction["question"]] = prediction

    all_predictions = existing_predictions
    i = 0
    time_list = []
    for example in tqdm(questions, total=len(questions), desc="Evaluation Progress"):

        question = example["question"]
        answers = example["answers"]
        print("-"*100)
        print(f"Question: {question}")
        print(f"Answers: {answers}")

        print("-"*100)
        if question in questions_that_have_predictions:
            print(f"Question {question} already has predictions, skipping ...")
            existing_prediction = questions_that_have_predictions[question]
            if long_form_qa:
                rouge_list.append(existing_prediction["rouge"])
            else:
                acc_list.append(existing_prediction["acc"])
                em_list.append(existing_prediction["em"])
                f1_list.append(existing_prediction["f1"])
            i += 1
            continue

        try:
            start_time = time.time()
            output: dict = model.forward_v2_for_reasoning_model(
                question=question,
                is_multihop_qa=multihop_qa,
                save_intermediate_results=args.save_intermediate_results,
                use_thought=False,
                is_long_form_qa=long_form_qa
            )
            time_list.append(time.time()-start_time)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"CUDA out of memory for question: {question}, skipping...")
                torch.cuda.empty_cache()
                i += 1
                continue
            else:
                raise e
        
        if long_form_qa:
            if model.use_long_reasoning_model:

                predicted_answer: str = output["output"].split("</think>")[-1].strip()
            else:
                predicted_answer: str = extract_non_reasoning_model_answer(output["output"])
            rouge = asqa_rouge_score(prediction=predicted_answer, ground_truths=answers)
            rouge_list.append(rouge)
            output["pred_answer"] = predicted_answer
            output["answers"] = answers
            output["rouge"] = rouge
        else:
            if model.use_long_reasoning_model:
                predicted_answer: str = extract_answer(output["output"], mode="qa")
            else:
                predicted_answer: str = extract_non_reasoning_model_answer(output["output"])

            acc = has_answer(answers, predicted_answer)
            em = ems(predicted_answer, answers)
            f1 = max(f1_score(predicted_answer, answer)[0] for answer in answers)
            acc_list.append(acc)
            em_list.append(em)
            f1_list.append(f1)

            output["pred_answer"] = predicted_answer
            output["answers"] = answers
            output["acc"] = acc
            output["em"] = em
            output["f1"] = f1

        all_predictions.append(output)

        i += 1
        if args.save_frequency is not None and i % args.save_frequency == 0:
            with open(os.path.join(args.save_dir, args.name, f"round_{round}_predictions.pkl"), "wb") as f:
                pickle.dump(all_predictions, f)

    avg_acc = float(np.mean(acc_list)) if acc_list else None
    avg_em = float(np.mean(em_list)) if em_list else None
    avg_f1 = float(np.mean(f1_list)) if f1_list else None
    avg_rouge = float(np.mean(rouge_list)) if rouge_list else None

    if time_list:
        print("Average Time: {}".format(np.mean(time_list)))


    with open(os.path.join(args.save_dir, args.name, f"round_{round}_predictions.pkl"), "wb") as f:
        pickle.dump(all_predictions, f)

    return {"acc": avg_acc, "em": avg_em, "f1": avg_f1} if avg_acc else {"rouge": avg_rouge}


if __name__ == "__main__":


    args = setup_parser()


    gpu_setup(args.local_rank, random_seed=42)


    checkpoint_path = os.path.join(args.save_dir, args.name)
    os.makedirs(checkpoint_path, exist_ok=True)
    setup_logger(args.local_rank, os.path.join(checkpoint_path, "results.log"))


    retriever_name = args.retriever.retriever_name
    if retriever_name == "OpenAITextEmbeddingRetriever":

        logger.info("loading collator class {} ...".format(args.dataset.collator_class_name))
        collator = COLLATOR_MAP[args.dataset.collator_class_name](**args.dataset.get_hparams())

        logger.info(f"loading retriever class {args.retriever.class_name}, model_name_or_path: {args.retriever.model_name_or_path} ... ")
        retriever = MODEL_MAP[args.retriever.class_name](local_rank=args.local_rank, **args.retriever.get_hparams())
    else:
        logger.info("loading tokenizer from {} ...".format(TOKENIZER_MAP[retriever_name]))
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MAP[retriever_name])
        if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
            logger.warning("Missing padding token, adding a new pad token!")
            tokenizer.add_special_tokens({"pad_token": '[PAD]'})


        logger.info("loading collator class {} ...".format(args.dataset.collator_class_name))
        collator = COLLATOR_MAP[args.dataset.collator_class_name](tokenizer=tokenizer, **args.dataset.get_hparams())


        logger.info(f"loading retriever class {args.retriever.class_name}, model_name_or_path: {args.retriever.model_name_or_path} ... ")
        retriever = MODEL_MAP[args.retriever.class_name](local_rank=args.local_rank, **args.retriever.get_hparams())
        retriever.to(device)
        retriever.eval()


    logger.info("loading corpus {} ... ".format(args.dataset.corpus))
    corpus_dataset = CORPUS_MAP[args.dataset.corpus](**args.dataset.get_hparams())


    logger.info(f"Loading index from {args.index_folder} ...")
    indexer = INDEX_MAP[args.index_name](args.embedding_size, metric="inner_product")
    indexer.deserialize_from(args.index_folder)


    zero_shot_dense_retriever = DenseRetriever(retriever=retriever, collator=collator, indexer=indexer, corpus=corpus_dataset, batch_size=args.per_gpu_batch_size)


    if retriever_name == "OpenAITextEmbeddingRetriever":
        logger.info("loading OpenRouter API KG Generator for KG Generation ...")
        kg_generator = APIKGGenerator(model_name_or_path="openai/gpt-4o-mini", examplar_type=args.dataset.name, **args.kg_generator.get_hparams())
    else:
        logger.info("loading llama3 tokenizer and model for KG Generation ...")
        llama3_tokenizer, llama3_model = load_llm_tokenizer_and_model("llama3", load_in_4bit=True, device_map="auto")
        kg_generator = KGGenerator(tokenizer=llama3_tokenizer, generator=llama3_model, examplar_type=args.dataset.name, **args.kg_generator.get_hparams())
    if args.cached_kg_triples_file is not None:
        kg_generator.load_cached_kg_triples(args.cached_kg_triples_file)


    kg_adaptive_rag = KGAdaptiveRAG(
        retriever=zero_shot_dense_retriever,
        kg_generator=kg_generator,
        **args.kg_adaptive_rag.get_hparams()
    )


    logger.info(f"loading query data from {args.query_file} ...")

    questions = load_json(args.query_file)
    if args.sample_k > 0:
        logger.info(f"sampling {args.sample_k} examples from the query file with seed 0 ...")
        seed_everything(0)
        questions = random.sample(questions, min(args.sample_k, len(questions)))


    all_round_results = []
    logger.info(f"Evaluating KG Adaptive RAG on {args.query_file} (sample_k={args.sample_k}) using {args.kg_adaptive_rag.reasoning_model_config.model_name_or_path} for {args.n_rounds} rounds ...")
    for i in range(args.n_rounds):
        logger.info(f"Evaluating for round {i+1} / {args.n_rounds} ...")











        results = qa_evaluate(
            args,
            questions = questions,
            model = kg_adaptive_rag,
            round = i+1,
        )
        logger.info(f"Round {i+1} results: {results}")
        all_round_results.append(results)


    average_results = {}
    for key in all_round_results[0].keys():
        average_results[key] = np.mean([results[key] for results in all_round_results])
    logger.info(f"Average results: {average_results}")

