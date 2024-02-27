import argparse
import json
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils.load_json import *
from models.agent_entity import *
from models.agent_relation import *
from models.agent_triplet import *
from models.agent_triplet_verifier import *
from models.agent_evaluation import *
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
        args.url = "http://localhost:8001"
        args.headers = {"Content-Type": "application/json"}

    if args.triplet_extracted:
        data = load_ndjson(DATA_PATH + str(args.dataset) + '.json')

        ############################################################################################################################################
        ############################################################################# Triplet extraction ###########################################
        ############################################################################################################################################
        print("stage 1: entity types and relation types preparation")
        # stage 1: entity types and relation types preparation
        ## entity types
        Ent_agent = Agent_entity(args)

        Ent_agent_prompt_all = []
        for index in range(len(data)):
            Ent_agent_prompt_all.append(Ent_agent.prompt_construction(data, index))

        Ent_agent.multi_threads_request(data, Ent_agent_prompt_all)

        ## relation types
        S_Rel_agent = Agent_spatial_relation(args)
        T_Rel_agent = Agent_temporal_relation(args)
        F_Rel_agent = Agent_functional_relation(args)

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


        print("stage 2: triplet extraction Chain-of-thought demonstration construction")
        # stage 2: triplet extraction chain-of-thought demonstration construction
        S_Triplet_agent_COT = Agent_spatial_triplet_COT_demonstration(args, args.demonstration_number)
        T_Triplet_agent_COT = Agent_temporal_triplet_COT_demonstration(args, args.demonstration_number)
        F_Triplet_agent_COT = Agent_funtional_triplet_COT_demonstration(args, args.demonstration_number)

        S_Triplet_agent_COT_prompt_all = []
        T_Triplet_agent_COT_prompt_all = []
        F_Triplet_agent_COT_prompt_all = []
        for index in range(args.demonstration_number):
            S_Triplet_agent_COT_prompt_all.append(
                S_Triplet_agent_COT.prompt_construction(data, index, Ent_agent.communication(index), S_Rel_agent.communication(index)))
            T_Triplet_agent_COT_prompt_all.append(
                T_Triplet_agent_COT.prompt_construction(data, index, Ent_agent.communication(index), T_Rel_agent.communication(index)))
            F_Triplet_agent_COT_prompt_all.append(
                F_Triplet_agent_COT.prompt_construction(data, index, Ent_agent.communication(index), F_Rel_agent.communication(index)))

        S_Triplet_agent_COT.multi_threads_request(data, S_Triplet_agent_COT_prompt_all)
        T_Triplet_agent_COT.multi_threads_request(data, T_Triplet_agent_COT_prompt_all)
        F_Triplet_agent_COT.multi_threads_request(data, F_Triplet_agent_COT_prompt_all)

        # 取出demonstration集合
        spatial_demonstration = []
        for i in range(args.used_demonstration_number):
            spatial_demonstration.append("Extract triplet related to spatial relationship from the following urban text: " +
                                         data[i]['text description'] +
                                         "Let's think step by step. " +
                                         S_Triplet_agent_COT.communication(i))

        temporal_demonstration = []
        for i in range(args.used_demonstration_number):
            temporal_demonstration.append("Extract triplet related to temporal relationship from the following urban text: " +
                                          data[i]['text description'] +
                                          "Let's think step by step. " +
                                          T_Triplet_agent_COT.communication(i))

        functional_demonstration = []
        for i in range(args.used_demonstration_number):
            functional_demonstration.append(
                                            "Extract triplet related to functional relationship from the following urban text: " +
                                            data[i]['text description'] +
                                            "Let's think step by step. " +
                                            F_Triplet_agent_COT.communication(i))

        print("stage 3: triplet extraction In-context-learning prompting")
        # stage 3: triplet extraction In-context-learning prompting
        S_Triplet_agent_ICL = Agent_spatial_triplet_ICL(args)
        T_Triplet_agent_ICL = Agent_temporal_triplet_ICL(args)
        F_Triplet_agent_ICL = Agent_functional_triplet_ICL(args)

        S_Triplet_agent_ICL_prompt_all = []
        T_Triplet_agent_ICL_prompt_all = []
        F_Triplet_agent_ICL_prompt_all = []
        for index in range(len(data)):
            S_Triplet_agent_ICL_prompt_all.append(
                S_Triplet_agent_ICL.prompt_construction(data, index, ''.join(spatial_demonstration)))
            T_Triplet_agent_ICL_prompt_all.append(
                T_Triplet_agent_ICL.prompt_construction(data, index, ''.join(temporal_demonstration)))
            F_Triplet_agent_ICL_prompt_all.append(
                F_Triplet_agent_ICL.prompt_construction(data, index, ''.join(functional_demonstration)))

        S_Triplet_agent_ICL.multi_threads_request(data, S_Triplet_agent_ICL_prompt_all)
        T_Triplet_agent_ICL.multi_threads_request(data, T_Triplet_agent_ICL_prompt_all)
        F_Triplet_agent_ICL.multi_threads_request(data, F_Triplet_agent_ICL_prompt_all)

        print("Formatting triplets from ICL prompting results")
        # formatting
        S_Triplet_from_ICL_agent = Agent_S_triplet_from_ICL(args)
        T_Triplet_from_ICL_agent = Agent_T_triplet_from_ICL(args)
        F_Triplet_from_ICL_agent = Agent_F_triplet_from_ICL(args)

        S_Triplet_from_ICL_agent_prompt_all = []
        T_Triplet_from_ICL_agent_prompt_all = []
        F_Triplet_from_ICL_agent_prompt_all = []
        for index in range(len(data)):

            S_Triplet_from_ICL_agent_prompt_all.append(S_Triplet_from_ICL_agent.prompt_construction(S_Triplet_agent_ICL.communication(index)))
            T_Triplet_from_ICL_agent_prompt_all.append(T_Triplet_from_ICL_agent.prompt_construction(T_Triplet_agent_ICL.communication(index)))
            F_Triplet_from_ICL_agent_prompt_all.append(F_Triplet_from_ICL_agent.prompt_construction(F_Triplet_agent_ICL.communication(index)))

        S_Triplet_from_ICL_agent.multi_threads_request(data, S_Triplet_from_ICL_agent_prompt_all)
        T_Triplet_from_ICL_agent.multi_threads_request(data, T_Triplet_from_ICL_agent_prompt_all)
        F_Triplet_from_ICL_agent.multi_threads_request(data, F_Triplet_from_ICL_agent_prompt_all)

        print("stage 4: iterative self-verification")
        # stage 4: iterative self-verification
        Triplet_verifier_agent = Agent_triplet_verifier(args)
        Triplet_updater_agent = Agent_triplet_updater(args)

        ## 给 updater的三元组进行初始化
        for index in range(len(data)):
            temp = str(S_Triplet_from_ICL_agent.communication(index)) + str(T_Triplet_from_ICL_agent.communication(index)) + str(F_Triplet_from_ICL_agent.communication(index))
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
                    Triplet_updater_agent_prompt_all.append(Triplet_updater_agent.prompt_construction(data, index, Triplet_updater_agent.communication(index), Triplet_verifier_agent.communication(index)))
                else:
                    Triplet_updater_agent_prompt_all.append({})

            Triplet_updater_agent.multi_threads_request(data, Triplet_updater_agent_prompt_all)

            print("Iteration:", epochs)
            if epochs >= args.max_iterative_epoch:
                break
            epochs += 1

        print("------Formatting -------")
        # formatting
        Triplet_format_agent = Agent_triplet_formatter(args)

        Triplet_format_agent_prompt_all = []
        for index in range(len(data)):

            Triplet_format_agent_prompt_all.append(Triplet_format_agent.prompt_construction(
                Triplet_updater_agent.communication(index)
            ))

        Triplet_format_agent.multi_threads_request(data, Triplet_format_agent_prompt_all)
        print('done')

    else:
        Triplet_format_agent = Agent_triplet_formatter(args)
        Triplet_format_agent.log_memories()

        ############################################################################################################################################
        ############################################################################# KG completion ################################################
        ############################################################################################################################################

        """
        {"entity ontology": "road",
         "entity name": "East 219th Street",
         "geometry type": "linestring",
         "entity type": "residential",
         "geometry value": "(-73.7461435 40.7647311, -73.7459859 40.7647422, -73.7458422 40.7647380, -73.7457377 40.7647194, -73.7456298 40.7647171)",
         "text description": "The 219th Street station is a local station on the IRT White Plains Road Line of the New York City Subway. Located at the intersection of 219th Street and White Plains Road in the Bronx, it is served by the 2 train at all times and by the 5 train during rush hours in the peak direction."
         "triplet": [{head entity: , relation: , tail entity: },
                     {head entity: , relation: , tail entity: },
                     ...
                     {head entity: , relation: , tail entity: },
                    ],
         }

        """
    data = load_ndjson(DATA_PATH + str(args.dataset) + '.json')
    extracted_triplet = Triplet_format_agent.memories
    # 首先将 每一条城市数据对应的三元组进行整合
    for index in range(len(data)):
        triplets = []

        extracted_triplet_str = extracted_triplet[index]['response'].replace("'", '"')
        try:
            extracted_triplet_list = json.loads(extracted_triplet_str)

            for k in range(len(extracted_triplet_list)):
                triplets_dict = {}
                try:
                    triplets_dict['head entity'] = extracted_triplet_list[k]['head entity']
                except:
                    triplets_dict['head entity'] = extracted_triplet_list[k]['head entity']
                triplets_dict['relation'] = extracted_triplet_list[k]['relation']
                try:
                    triplets_dict['tail entity'] = extracted_triplet_list[k]['tail entity']
                except:
                    triplets_dict['tail entity'] = extracted_triplet_list[k]['tail entity']
                triplets.append(triplets_dict)
        except:
            extracted_triplet_str = str2triplet(extracted_triplet_str)
            triplets.append(extracted_triplet_str)

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









