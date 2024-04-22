from models.base import KGCAgent_base
import threading
import requests
import json
from tqdm import tqdm
import copy

class KGCAgent_Trajectory(KGCAgent_base):

    def __init__(self, args):

        super(KGCAgent_Trajectory, self).__init__(args.url, args.headers, args.model, args.temperature, args.tokens, args.threads, args.dataset)
        self.url = self.url
        self.headers = self.headers
        self.model = self.model
        self.temperature = self.temperature
        self.tokens = self.tokens
        self.threads = self.threads
        self.dataset = self.dataset
        self.memories = []

class KGCAgent_Trajectory_Verifier(KGCAgent_Trajectory):
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

    def multi_threads_request(self, input, prompt_completion_all):

        threads = []
        for index in tqdm(range(len(prompt_completion_all))):
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

    def prompt_construction(self, input, index, reasoning_process):

        head_lat_lng, tail_lat_lng = self.get_lat_lng(input, index)

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "Here are the reasoning process when using region connection calculus (RCC) to describes the geospatial relationships between the two geospatial entities."
                            "RCC consists of 5 basic relations that are possible between two geospatial entities: (1) Disconnected (DC); (2) Externally connected (EC); (3) Equal (EQ); (4) Partially Overlapping (PO); (5) Tangential and nontangential proper parts (IN)."
                            "Based on the above reasoning process, please justify: 1) whether constructed RCC8 relationship are correct."
                            "If constructed RCC8 relation is coorect, please answer 'Yes'. If not, please provide improvement suggestions to help better identify the RCC8 relation between these two geospatial entities." + '\n'},
                {"role": "user", "content": "Entity 1: Latitude and Longitude: " + head_lat_lng},
                {"role": "user", "content": "Entity 2: Latitude and Longitude: " + tail_lat_lng + '\n'},
                {"role": "user", "content": "Reasoning process:" + reasoning_process + "\n"},
                {"role": "user", "content": "Return the result with the following format: {\"Answer\": \"Yes/No\", \"Improvement suggestions\": \"Suggestion\"}"},

            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }

        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion


class KGCAgent_Trajectory_Updater(KGCAgent_Trajectory):

    def multi_threads_request(self, input, prompt_completion_all):
        threads = []
        for index in tqdm(range(len(prompt_completion_all))):
            if len(prompt_completion_all[index]) == 0:
                pass
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


    def prompt_construction(self, input, index, reasoning_process, improvement):
        head_lat_lng, tail_lat_lng = self.get_lat_lng(input, index)

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "Here are the reasoning process when using region connection calculus (RCC) to describes the geospatial relationships between the two geospatial entities. "
                            "RCC consists of 5 basic relations that are possible between two geospatial entities: (1) Disconnected (DC); (2) Externally connected (EC); (3) Equal (EQ); (4) Partially Overlapping (PO); (5) Tangential and nontangential proper parts (IN)."
                            "Please follow the improvement suggestion to refine extracted RCC8 relations." + '\n'},
                {"role": "user", "content": "Entity 1: Latitude and Longitude: " + head_lat_lng},
                {"role": "user", "content": "Entity 2: Latitude and Longitude: " + tail_lat_lng + '\n'},
                {"role": "user", "content": "Reasoning process:" + reasoning_process + '\n'},
                {"role": "user", "content": "Improvement Suggestion:" + improvement + '\n'},
                {"role": "user", "content": "Return the result with the following format: {\"Geospatial relation\": \"DC\"}"},

            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion

class KGCAgent_Trajectory_Formatter(KGCAgent_Trajectory):

    def prompt_construction(self, reasoning_process):
        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "RCC consists of 5 basic relations that are possible between two geospatial entities: (1) Disconnected (DC); (2) Externally connected (EC); (3) Equal (EQ); (4) Partially Overlapping (PO); (5) Tangential and nontangential proper parts (IN)." + '\n'},
                {"role": "user",
                 "content": "Here are the reasoning process when using region connection calculus (RCC) to describes the geospatial relationships between the two geospatial entities."
                            "Based on the above reasoning process, please directly output which RCC8 relationship these two geospatial entities belong to. Output the answer without any other textual description." + '\n'},
                {"role": "user", "content": "Reasoning process:" + reasoning_process + "\n" },
                {"role": "user",
                 "content": "Return the result with the following format without any other explanation: {\"Geospatial relation\": \"DC\"}"},

            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion