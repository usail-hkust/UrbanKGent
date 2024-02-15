from models.base import TEAgent_base
import threading
import requests
import json
from tqdm import tqdm
import copy
import string

class Agent_triplet_verifier(TEAgent_base):

    def __init__(self, args):

        super(Agent_triplet_verifier, self).__init__(args.url, args.headers, args.model, args.temperature, args.tokens, args.threads, args.dataset)
        self.url = self.url
        self.headers = self.headers
        self.model = self.model
        self.temperature = self.temperature
        self.tokens = self.tokens
        self.threads = self.threads
        self.dataset = self.dataset
        self.memories = []

    def request(self, input, prompt_completion, index):
        flag = False
        while flag is not True:
            try:
                response = requests.post(self.url, headers=self.headers, data=json.dumps(prompt_completion))
                response_json = response.json()
                if "llama" in self.model:
                    results = response_json['response']
                else:
                    results = response_json['choices'][0]['message']['content']
                if "yes" in results.lower():
                    state = "Yes"
                else:
                    state = results

                # 写入 memory
                self.memory(input, index, prompt_completion, state)

                flag = True
            except Exception as e:
                flag = False

    def prompt_construction(self, input, index, triplet):

        text = input[index]['text description']
        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "Given the urban text data and the extracted triplets, please justify: 1) whether all extracted triplets are correct; 2) whether there are missing triplets that have not yet been extracted."
                                                "If there are no missing triplets and all triplets are coorect, please answer 'Yes'. If not, please provide improvement suggestions to help extract missing triplet and remove the incorrect triplets." + '\n'},
                    {"role": "user", "content": "Urban text:" + text + '\n'},
                    {"role": "user", "content": "Extracted triplet:" + triplet},
                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion

class Agent_triplet_updater(TEAgent_base):
    def __init__(self, args):
        super(Agent_triplet_updater, self).__init__(args.url, args.headers, args.model, args.temperature, args.tokens, args.threads, args.dataset)
        self.url = self.url
        self.headers = self.headers
        self.model = self.model
        self.temperature = self.temperature
        self.tokens = self.tokens
        self.threads = self.threads
        self.dataset = self.dataset
        self.memories = []

    def multi_threads_request(self, input, prompt_completion_all):
        threads = []
        for index in tqdm(range(len(prompt_completion_all))):
            if len(prompt_completion_all[index]) == 0:
                continue
            else:
                t = threading.Thread(target=self.request, args=(input, prompt_completion_all[index], index,))
                threads.append(t)
                while len(threads) == self.threads or index == len(prompt_completion_all) - 1:
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                        # print("Thread", t.ident, "has finished")
                    threads = []
                    break

        self.log()

    def prompt_construction(self, input, index, triplet, improvement):

        text = input[index]['text description']
        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "Given the urban text data and the extracted triplets, please follow the suggestion to remove incorrect triplet or add missing triplet." + '\n'},
                {"role": "user", "content": "Urban text:" + text + '\n'},
                {"role": "user", "content": "Extracted triplet:" + triplet + '\n'},
                {"role": "user", "content": "Suggestion for improvement:" + improvement + '\n'},
                {"role": "user", "content": "Return the result with the {'head entity': ' ', 'relation': ' ', 'tail entity': ' '} format"},
            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion


class Agent_triplet_from_ICL(TEAgent_base):
    def __init__(self, args):

        super(Agent_triplet_from_ICL, self).__init__(args.url, args.headers, args.model, args.temperature, args.tokens, args.threads, args.dataset)
        self.url = self.url
        self.headers = self.headers
        self.model = self.model
        self.temperature = self.temperature
        self.tokens = self.tokens
        self.threads = self.threads
        self.dataset = self.dataset
        self.memories = []

class Agent_ablation_triplet_from_ICL(Agent_triplet_from_ICL):

    def prompt_construction(self, S_triplet):

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "Here are the reasoning process and extracted triplets. Please filter the reasoning process and just return the extracted triplets. Return the result with the {'head entity': ' ', 'relation': ' ', 'tail entity': ' '} format." + '\n'},
                    {"role": "user", "content": "Reasoning process and extracted triplets:" + S_triplet},

                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion


class Agent_S_triplet_from_ICL(Agent_triplet_from_ICL):

    def prompt_construction(self, S_triplet):

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "Here are the reasoning process and extracted triplets. Please filter the reasoning process and just return the extracted triplets. Return the result with the {'head entity': ' ', 'relation': ' ', 'tail entity': ' '} format." + '\n'},
                    {"role": "user", "content": "Reasoning process and extracted triplets:" + S_triplet},

                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion

class Agent_T_triplet_from_ICL(Agent_triplet_from_ICL):

    def prompt_construction(self, T_triplet):

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "Here are the reasoning process and extracted triplets. Please filter the reasoning process and just return the extracted triplets. Return the result with the {'head entity': ' ', 'relation': ' ', 'tail entity': ' '} format." + '\n'},
                    {"role": "user", "content": "Reasoning process and extracted triplets:" + T_triplet},

                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion

class Agent_F_triplet_from_ICL(Agent_triplet_from_ICL):

    def prompt_construction(self, F_triplet):

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "Here are the reasoning process and extracted triplets. Please filter the reasoning process and just return the extracted triplets. Return the result with the {'head entity': ' ', 'relation': ' ', 'tail entity': ' '} format." + '\n'},
                    {"role": "user", "content": "Reasoning process and extracted triplets:" + F_triplet},

                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion

class Agent_triplet_formatter(TEAgent_base):
    def __init__(self, args):

        super(Agent_triplet_formatter, self).__init__(args.url, args.headers, args.model, args.temperature, args.tokens, args.threads, args.dataset)
        self.url = self.url
        self.headers = self.headers
        self.model = self.model
        self.temperature = self.temperature
        self.tokens = self.tokens
        self.threads = self.threads
        self.dataset = self.dataset
        self.memories = []

    def prompt_construction(self, triplet):

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "Organize the provided triplet into the following format: {'head entity': ' ', 'relation': ' ', 'tail entity': ' '}."
                                                "Return the results without any textual description." + '\n'},
                    {"role": "user", "content": "Triplets:" + triplet}

                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion

