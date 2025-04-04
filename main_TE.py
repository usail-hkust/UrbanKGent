import argparse
import json
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import re
from utils.load_json import *
from models.RTE_Instruction_Generator import *
from models.RTE_Trajectory_Refinement import *
from models.Evaluator import *
from utils.acc_confidence_calculate import *
from utils.triplet_formatting import *
import copy
from tqdm import tqdm
import random

DATA_PATH = './data/TE_data/'
RESULT_PATH = './log/TE_log/'

parser = argparse.ArgumentParser(
    description="Urban Knowledge Graph Construction"
)
parser.add_argument(
    "--dataset", default="debug", choices=["NYC", "CHI", "NYC_Instruct", "CHI_Instruct", "debug"],
    help="Urban Knowledge Graph construction datasets"
)
parser.add_argument(
    "--url", default="https://gpt-api.hkust-gz.edu.cn/v1/chat/completions", type=str,
    choices=["https://gpt-api.hkust-gz.edu.cn/v1/chat/completions", "http://localhost:8000"],
    help="first for GPT-API, the second for Llama"
)
parser.add_argument(
    "--headers", default={"Content-Type": "application/json", "Authorization": "Bearer api-key"},
    choices=[{"Content-Type": "application/json", "Authorization": "Bearer Your API-Key"},
             {"Content-Type": "application/json"}
             ],
    help="first for GPT-API, the second for Llama"
)
parser.add_argument(
    "--model", default="gpt-4", type=str, choices=["gpt-3.5-turbo", 'gpt-4', "gpt-4-32k", "llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "llama-2-70b-chat-hf"]
)
parser.add_argument(
    "--temperature", default=0.2, choices=[0.25, 0.5, 0.75, 1]
)
parser.add_argument(
    "--tokens", default=32000, choices=[512, 8000, 32000]
)
parser.add_argument(
    "--evaluation_url", default="https://gpt-api.hkust-gz.edu.cn/v1/chat/completions", type=str
)
parser.add_argument(
    "--evaluation_headers", default={"Content-Type": "application/json", "Authorization": "Bearer Your api-key"},
)
parser.add_argument(
    "--evaluation_model", default="gpt-4", type=str, choices=["llama-2-13b-finetune", "llama-2-7b-finetune", "gpt-3.5-turbo", "text-davinci-003", 'gpt-4', "gpt-4-32k", "llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "llama-2-70b-chat-hf"]
)
parser.add_argument(
    "--evaluation_tokens", default=32000, choices=[512, 8000, 32000]
)
parser.add_argument(
    "--threads", default=1, choices=[10, 20, 40, 50]
)
parser.add_argument(
    "--demonstration_number", default=1, choices=[2, 3, 4, 5]
)
parser.add_argument(
    "--used_demonstration_number", default=1, choices=[2, 3, 4, 5]
)
parser.add_argument(
    "--max_iterative_epoch", default=3,  choices=[2, 5, 10]
)

if __name__ == "__main__":
    args = parser.parse_args()
    if "llama" in args.model:
        args.url = "http://localhost:8000"
        args.headers = {"Content-Type": "application/json"}

    data = load_ndjson(DATA_PATH + str(args.dataset) + '.json')

    ############################################################################################################################################
    ############################################################################# Triplet extraction ###########################################
    ############################################################################################################################################
    print("stage 1: entity types and relation types preparation")
    # stage 1: entity types and relation types preparation
    ## entity types
    S_Ent_agent = RTEAgent_NER_Spatial(args)
    T_Ent_agent = RTEAgent_NER_Temporal(args)
    F_Ent_agent = RTEAgent_NER_Functional(args)

    S_Ent_agent_prompt_all = []
    T_Ent_agent_prompt_all = []
    F_Ent_agent_prompt_all = []
    for index in range(len(data)):
        S_Ent_agent_prompt_all.append(S_Ent_agent.prompt_construction(data, index))
        T_Ent_agent_prompt_all.append(T_Ent_agent.prompt_construction(data, index))
        F_Ent_agent_prompt_all.append(F_Ent_agent.prompt_construction(data, index))

    S_Ent_agent.multi_threads_request(data, S_Ent_agent_prompt_all)
    T_Ent_agent.multi_threads_request(data, T_Ent_agent_prompt_all)
    F_Ent_agent.multi_threads_request(data, F_Ent_agent_prompt_all)

    ## relation types
    S_Rel_agent = RTEAgent_RE_Spatial(args)
    T_Rel_agent = RTEAgent_RE_Temporal(args)
    F_Rel_agent = RTEAgent_RE_Functional(args)

    S_Rel_agent_prompt_all = []
    T_Rel_agent_prompt_all = []
    F_Rel_agent_prompt_all = []

    for index in range(len(data)):
        S_Rel_agent_prompt_all.append(S_Rel_agent.prompt_construction(data, index))
        T_Rel_agent_prompt_all.append(T_Rel_agent.prompt_construction(data, index))
        F_Rel_agent_prompt_all.append(F_Rel_agent.prompt_construction(data, index))

    S_Rel_agent.multi_threads_request(data, S_Rel_agent_prompt_all)
    T_Rel_agent.multi_threads_request(data, T_Rel_agent_prompt_all)
    F_Rel_agent.multi_threads_request(data, F_Rel_agent_prompt_all)


    print("stage 2: triplet extraction")
    # stage 2: triplet extraction chain-of-thought demonstration construction
    S_Triplet_agent = RTEAgent_TE_Spatial(args)
    T_Triplet_agent = RTEAgent_TE_Temporal(args)
    F_Triplet_agent = RTEAgent_TE_Functional(args)

    S_Triplet_agent_COT_prompt_all = []
    T_Triplet_agent_COT_prompt_all = []
    F_Triplet_agent_COT_prompt_all = []
    for index in range(len(data)):
        S_Triplet_agent_COT_prompt_all.append(
            S_Triplet_agent.prompt_construction(data, index, S_Ent_agent.communication(index), S_Rel_agent.communication(index)))
        T_Triplet_agent_COT_prompt_all.append(
            T_Triplet_agent.prompt_construction(data, index, T_Ent_agent.communication(index), T_Rel_agent.communication(index)))
        F_Triplet_agent_COT_prompt_all.append(
            F_Triplet_agent.prompt_construction(data, index, F_Ent_agent.communication(index), F_Rel_agent.communication(index)))

    S_Triplet_agent.multi_threads_request(data, S_Triplet_agent_COT_prompt_all)
    T_Triplet_agent.multi_threads_request(data, T_Triplet_agent_COT_prompt_all)
    F_Triplet_agent.multi_threads_request(data, F_Triplet_agent_COT_prompt_all)

    print("stage 3: iterative self-verification")
    # stage 4: iterative self-verification
    Triplet_verifier_agent = RTEAgent_Trajectory_Verifier(args)
    Triplet_updater_agent = RTEAgent_Trajectory_Updater(args)

    ## 给 updater的三元组进行初始化
    for index in range(len(data)):
        pattern_s = r"Spatial triplet 1(.*)"
        matche_s = re.search(pattern_s, S_Triplet_agent.communication(index), re.IGNORECASE | re.DOTALL)
        if matche_s == None:
            matche_s = ''
        pattern_t = r"Temporal triplet 1(.*)"
        matche_t = re.search(pattern_t, T_Triplet_agent.communication(index), re.IGNORECASE | re.DOTALL)
        if matche_t == None:
            matche_t = 'none'
        pattern_f = r"Functional triplet 1(.*)"
        matche_f = re.search(pattern_f, F_Triplet_agent.communication(index), re.IGNORECASE | re.DOTALL)
        if matche_f == None:
            matche_f = ''

        temp = str(matche_s[0]) + str(matche_t[0]) + str(matche_f[0])
        data[index]['prompt_completion'] = ' '
        data[index]['response'] = temp
        # The memory for Entity Agent
        Triplet_updater_agent.memories.append(copy.deepcopy(data)[index])

    # self-verify and update until max epoch or all triplet are yes
    all_finished = False
    epochs = 1
    while all_finished == False:
        all_finished = True

        # verify whether all triplet are correct and there no more missing triplet
        Triplet_verifier_agent_prompt_all = []
        for index in range(len(data)):
            Triplet_verifier_agent_prompt_all.append(Triplet_verifier_agent.prompt_construction(data, index, Triplet_updater_agent.communication(index)))

        Triplet_verifier_agent.multi_threads_request(data, Triplet_verifier_agent_prompt_all)

        # tag
        Triplet_updater_agent_prompt_all = []
        for index in range(len(data)):
            tag = Triplet_verifier_agent.communication(index)
            if "yes" not in tag.lower():
                all_finished = False
                # update triplet based on suggestion
                pattern = r"Improvement suggestions(.*)"
                matche = re.search(pattern, Triplet_verifier_agent.communication(index), re.DOTALL)
                if matche == None:
                    improve_sug = Triplet_verifier_agent.communication(index)
                else:
                    improve_sug = matche[0]
                Triplet_updater_agent_prompt_all.append(Triplet_updater_agent.prompt_construction(data, index, Triplet_updater_agent.communication(index), improve_sug))
            else:
                Triplet_updater_agent_prompt_all.append({})

        Triplet_updater_agent.multi_threads_request(data, Triplet_updater_agent_prompt_all)

        print("Iteration:", epochs)
        if epochs >= args.max_iterative_epoch:
            break
        epochs += 1

    print("------Formatting -------")
    # formatting
    Triplet_format_agent = RTEAgent_Trajectory_Formatter(args)

    Triplet_format_agent_prompt_all = []
    for index in range(len(data)):

        Triplet_format_agent_prompt_all.append(Triplet_format_agent.prompt_construction(
            Triplet_updater_agent.communication(index)
        ))

    Triplet_format_agent.multi_threads_request(data, Triplet_format_agent_prompt_all)
    print('done')

    data = load_ndjson(DATA_PATH + str(args.dataset) + '.json')
    Extracted_triplet = Triplet_format_agent.memories
    # 首先将 每一条城市数据对应的三元组进行整合
    for index in range(len(data)):
        triplets = []
        pattern__ = r"{\"Triplet 1\"(.*?)\"}}"
        matche_ = re.search(pattern__, Extracted_triplet[index]['response'], re.IGNORECASE | re.DOTALL)
        if matche_ ==  None:
            triplet_json = Extracted_triplet[index]['response']
            triplets.append(triplet_json)
        else:
            triplet_json = matche_[0]
            extracted_triplet = json.loads(triplet_json)
            for key in extracted_triplet.keys():
                triplets.append(extracted_triplet[key])

        data[index]['triplet'] = triplets

    #### Triplet evaluation
    Evaluation_agent = Agent_evaluation_TE(args)

    Evaluation_agent_prompt_all = []
    for index in range(len(data)):
        Evaluation_agent_prompt_all.append(Evaluation_agent.prompt_construction(data, index))

    Evaluation_agent.multi_threads_request(data, Evaluation_agent_prompt_all)

    # 结果写入
    results = Evaluation_agent.memories
    with open(RESULT_PATH + args.dataset + '_' + str(args.model) + '_triplet.json', 'w') as f:
        for dic in data:
            f.write(json.dumps(dic) + '\n')
    print('done')

    print('Calculating the accuracy of thr results:')
    accuracy, avg_confidence = TE_acc_confidence(RESULT_PATH + args.dataset + '_' + str(args.model) + '_triplet.json')
    print('accuracy:', accuracy)
    print('confidence:', avg_confidence)









