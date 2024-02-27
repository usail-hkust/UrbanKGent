import argparse
import json
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import copy
from tqdm import tqdm
import random

from utils.load_json import *
from models.agent_geocompletion import *
from models.agent_geocompletion_verifier import *
from models.agent_evaluation import *
from utils.acc_confidence_calculate import *

DATA_PATH = './data/KGC_data/'
RESULT_PATH = './log/KGC_log/'

parser = argparse.ArgumentParser(
    description="Urban Knowledge Graph Construction"
)
parser.add_argument(
    "--dataset", default="NYC", choices=["NYC", "CHI", "NYC_Instruct", "CHI_Instruct"],
    help="Urban Knowledge Graph construction datasets"
)
parser.add_argument(
    "--url", default="https://gpt-api.hkust-gz.edu.cn/v1/chat/completions", type=str,
    choices=["https://gpt-api.hkust-gz.edu.cn/v1/chat/completions", "http://localhost:8000"],
    help="first for GPT-API, the second for Llama"
)
parser.add_argument(
    "--headers", default={"Content-Type": "application/json", "Authorization": "Bearer Your API-Key"},
    choices=[{"Content-Type": "application/json", "Authorization": "Bearer Your API-Key"},
             {"Content-Type": "application/json"}
             ],
    help="first for GPT-API, the second for Llama"
)
parser.add_argument(
    "--model", default="llama-2-7b-chat-hf", type=str, choices=["gpt-3.5-turbo", 'gpt-4', "gpt-4-32k", "llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "llama-2-70b-chat-hf"]
)
parser.add_argument(
    "--temperature", default=0, choices=[0.25, 0.5, 0.75, 1]
)
parser.add_argument(
    "--tokens", default=8000, choices=[512, 8000, 32000]
)
parser.add_argument(
    "--evaluation_url", default="https://gpt-api.hkust-gz.edu.cn/v1/chat/completions", type=str
)
parser.add_argument(
    "--evaluation_headers", default={"Content-Type": "application/json", "Authorization": "Bearer Your API-Key"},
)
parser.add_argument(
    "--evaluation_model", default="gpt-3.5-turbo", type=str, choices=["llama-2-13b-finetune", "llama-2-7b-finetune", "gpt-3.5-turbo", "text-davinci-003", 'gpt-4', "gpt-4-32k", "llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "llama-2-70b-chat-hf"]
)
parser.add_argument(
    "--evaluation_tokens", default=32000, choices=[512, 8000, 32000]
)
parser.add_argument(
    "--threads", default=20, choices=[10, 20, 40, 50]
)
parser.add_argument(
    "--demonstration_number", default=3, choices=[2, 3, 4, 5]
)
parser.add_argument(
    "--used_demonstration_number", default=1, choices=[2, 3, 4, 5]
)
parser.add_argument(
    "--max_iterative_epoch", default=3,  choices=[2, 5, 10]
)
parser.add_argument(
    "--triplet_extracted", default=True, choices=[True, False]
)

if __name__ == "__main__":
    args = parser.parse_args()

    if "llama" in args.model:
        args.url = "http://localhost:8000"
        args.headers = {"Content-Type": "application/json"}

    data = load_ndjson(DATA_PATH + str(args.dataset) + '.json')

    print("step 1: COT")
    # step 1: geo hash, COT geo KGC
    GEO_COT_agent = GEO_RCC8_COT_Agent(args)

    GEO_agent_prompt_all = []
    for index in range(len(data)):
        GEO_agent_prompt_all.append(GEO_COT_agent.prompt_construction(data, index))

    GEO_COT_agent.multi_threads_request(data, GEO_agent_prompt_all)

    print("step 2: call for tool interface")
    GEO_RCC8_Tool_agent = GEO_RCC8_Tool_Agent(args)

    GEO_RCC8_Tool_agent_prompt_all = []
    for index in range(len(data)):
        GEO_RCC8_Tool_agent_prompt_all.append(GEO_RCC8_Tool_agent.prompt_construction(data, index, GEO_COT_agent.communication(index)))

    GEO_RCC8_Tool_agent.multi_threads_request(data, GEO_RCC8_Tool_agent_prompt_all)

    print("step 3: Tool-enhanced COT")
    # step 2: ICL geo KGC;
    GEO_RCC8_Tool_COT_agent = GEO_RCC8_Tool_COT_Agent(args)

    GEO_RCC8_Tool_COT_agent_prompt_all = []
    for index in range(len(data)):
        GEO_RCC8_Tool_COT_agent_prompt_all.append(GEO_RCC8_Tool_COT_agent.prompt_construction(data, index,GEO_COT_agent.communication(index), GEO_RCC8_Tool_agent.communication(index)))

    GEO_RCC8_Tool_COT_agent.multi_threads_request(data, GEO_RCC8_Tool_COT_agent_prompt_all)


    print("step 3: self-verifying")
    # step 3: self-verifying
    GEOKGC_verifier_agent = GEOKGC_Agent_Verifier(args)
    GEOKGC_updater_agent = GEOKGC_Agent_Updater(args)

    ## 给 updater的三元组进行初始化
    for index in range(len(data)):
        temp = str(GEO_RCC8_Tool_COT_agent.communication(index))
        data[index]['prompt_completion'] = ' '
        data[index]['response'] = temp
        # The memory for Entity Agent
        GEOKGC_updater_agent.memories.append(copy.deepcopy(data)[index])

    # self-verify and update until max epoch or all triplet are yes
    all_finished = False
    epochs = 1
    while all_finished == False:
        all_finished = True

        # verify whether all triplet are correct and there no more missing triplet
        GEOKGC_verifier_agent_prompt_all = []
        for index in range(len(data)):
            GEOKGC_verifier_agent_prompt_all.append(
                GEOKGC_verifier_agent.prompt_construction(data, index, GEOKGC_updater_agent.communication(index)))

        GEOKGC_verifier_agent.multi_threads_request(data, GEOKGC_verifier_agent_prompt_all)
        # tag
        GEOKGC_updater_agent_prompt_all = []
        for index in range(len(data)):
            tag = GEOKGC_verifier_agent.communication(index)
            if "yes" not in tag.lower():
                all_finished = False
                # update triplet based on suggestion
                GEOKGC_updater_agent_prompt_all.append(GEOKGC_updater_agent.prompt_construction(data, index,
                                                                                                GEOKGC_updater_agent.communication(index),
                                                                                                GEOKGC_verifier_agent.communication(index)))
            else:
                GEOKGC_updater_agent_prompt_all.append({})

        GEOKGC_updater_agent.multi_threads_request(data, GEOKGC_updater_agent_prompt_all)

        print("Iteration:", epochs)
        if epochs >= args.max_iterative_epoch:
            break
        epochs += 1

    print("Formatting triplets from prompting results")
    # formatting
    GEO_RCC8_ICL_Format_agent = GEO_RCC8_ICL_Format_Agent(args)

    GEO_RCC8_ICL_Format_agent_prompt_all = []

    for index in range(len(data)):
        GEO_RCC8_ICL_Format_agent_prompt_all.append(
            GEO_RCC8_ICL_Format_agent.prompt_construction(GEOKGC_updater_agent.communication(index)))

    GEO_RCC8_ICL_Format_agent.multi_threads_request(data, GEO_RCC8_ICL_Format_agent_prompt_all)

    print('Evaluation')
    data = load_ndjson(DATA_PATH + str(args.dataset) + '.json')
    for index in range(len(data)):
        data[index]['Geo relation'] = GEO_RCC8_ICL_Format_agent.communication(index)

    #### Triplet evaluation
    Evaluation_agent = Agent_evaluation_KGC(args)

    Evaluation_agent_prompt_all = []
    for index in range(len(data)):
        Evaluation_agent_prompt_all.append(Evaluation_agent.prompt_construction(data, index))

    Evaluation_agent.multi_threads_request(data, Evaluation_agent_prompt_all)

    # 结果写入
    results = Evaluation_agent.memories
    with open(RESULT_PATH + args.dataset + '_' + str(args.model) + '_georelation.json', 'w') as f:
        for dic in data:
            f.write(json.dumps(dic) + '\n')

    print('Calculating the accuracy of thr results:')
    accuracy, avg_confidence = KGC_acc_confidence(RESULT_PATH + args.dataset + '_' + str(args.model) + '_georelation.json')
    print('accuracy:', accuracy)
    print('confidence:', avg_confidence)

