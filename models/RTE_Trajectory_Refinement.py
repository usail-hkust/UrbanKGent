from models.base import RTEAgent_base
import threading
import requests
import json
from tqdm import tqdm
import copy
import string

class RTEAgent_Trajectory(RTEAgent_base):

    def __init__(self, args):

        super(RTEAgent_Trajectory, self).__init__(args.url, args.headers, args.model, args.temperature, args.tokens, args.threads, args.dataset)
        self.url = self.url
        self.headers = self.headers
        self.model = self.model
        self.temperature = self.temperature
        self.tokens = self.tokens
        self.threads = self.threads
        self.dataset = self.dataset
        self.memories = []

class RTEAgent_Trajectory_Verifier(RTEAgent_Trajectory):

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
                    {"role": "user", "content": "Extracted triplet:" + triplet + '\n'},
                    {"role": "user", "content": "Return the result with the following format: {\"Answer\": \"Yes/No\", \"Improvement suggestions\": \"Suggestion\"}"},
                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion


class RTEAgent_Trajectory_Updater(RTEAgent_Trajectory):

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
                {"role": "user", "content": "Return the result with the following format: {\"Triplet 1\": {\"Head entity\": \"\", \"Relation\": \"\", \"Tail entity\": \"\"}, \"Triplet 2\": {\"Head entity\": \"\", \"Relation\": \"\", \"Tail entity\": \"\"}}"},
            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion

class RTEAgent_Trajectory_Formatter(RTEAgent_Trajectory):

    def prompt_construction(self, triplet):

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "Please help organize the provided spatial temporal and functional triplet into an unified format." + '\n'},
                    {"role": "user", "content": "Triplets:" + triplet + '\n'},
                    {"role": "user", "content": "Return the result with the following format without any other explanation: {\"Triplet 1\": {\"Head entity\": \"\", \"Relation\": \"\", \"Tail entity\": \"\"}, \"Triplet 2\": {\"Head entity\": \"\", \"Relation\": \"\", \"Tail entity\": \"\"}}"},
                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion