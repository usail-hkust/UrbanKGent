from abc import abstractmethod
import threading
import requests
import json
from tqdm import tqdm
from utils.load_json import *
from utils.tool_interface import *
from utils.prompt_transfer import prompt_completion_transfer
import copy
import os

class RTEAgent_base:
    def __init__(self, url, headers, model, temperature, tokens, threads, dataset):
        self.url = url
        self.headers = headers
        self.model = model
        self.temperature = temperature
        self.tokens = tokens
        self.threads = threads
        self.dataset = dataset
        self.memories = []

    @abstractmethod
    def prompt_construction(self, input):
        """
        construct the prompt completion for Agent

        :param input:
        :return:
        """
        pass

    def prompt_gpt_to_llama(self, prompt_completion):
        """
        construct the prompt completion for Agent

        :param input:
        :return:
        """
        return prompt_completion_transfer(prompt_completion)

    def memory(self, input, index, prompt_completion, response):
        input[index]['prompt_completion'] = prompt_completion
        input[index]['response'] = response

        # The memory for Entity Agent
        self.memories.append(copy.deepcopy(input)[index])

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
                # 写入 memory
                self.memory(input, index, prompt_completion, results)

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

    def communication(self, index):
        """
        返回抽取出来的 entity types

        :param instruction:
        :return:
        """
        return self.memories[index]['response']

    def log(self):
        """
        将agent执行结束后的prompt_completion和response记录，在multi_threads_request之后进行记录
        :return:
        """
        save_dir_log = './prompt/TE/' + str(self.model) + '/'
        if not os.path.exists(save_dir_log):
            try:
                os.makedirs(save_dir_log)
            except OSError as e:
                print(f"Creating dir'{save_dir_log}': {e}")
        else:
            print(f"Dir '{save_dir_log}' existing!")

        save_log = './prompt/TE/' + str(self.model) + '/' + str(self.dataset) + '_'+ str(self.__class__.__name__) \
                   + '_' + str(self.temperature) + '_' + str(self.tokens) + '_' + str(self.threads) + '.json'
        with open(save_log, 'w') as f:
            for dic in self.memories:
                f.write(json.dumps(dic) + '\n')

    def log_memories(self):
        save_log = './prompt/TE/' + str(self.model) + '/' + str(self.dataset) + '_' + str(self.__class__.__name__) \
                   + '_' + str(self.temperature) + '_' + str(self.tokens) + '_' + str(self.threads) + '.json'

        self.memories = load_ndjson(save_log)

class KGCAgent_base:
    def __init__(self, url, headers, model, temperature, tokens, threads, dataset):
        self.url = url
        self.headers = headers
        self.model = model
        self.temperature = temperature
        self.tokens = tokens
        self.threads = threads
        self.dataset = dataset
        self.memories = []

    @abstractmethod
    def prompt_construction(self, input):
        """
        construct the prompt completion for Agent

        :param input:
        :return:
        """
        pass

    def prompt_gpt_to_llama(self, prompt_completion):
        """
        construct the prompt completion for Agent

        :param input:
        :return:
        """
        return prompt_completion_transfer(prompt_completion)

    def memory(self, input, index, prompt_completion, response):
        input[index]['prompt_completion'] = prompt_completion
        input[index]['response'] = response

        # The memory for Entity Agent
        self.memories.append(copy.deepcopy(input)[index])

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

                # 写入 memory
                self.memory(input, index, prompt_completion, results)

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

    def communication(self, index):
        """
        返回抽取出来的 entity types

        :param instruction:
        :return:
        """
        return self.memories[index]['response']

    def log(self):
        """
        将agent执行结束后的prompt_completion和response记录，在multi_threads_request之后进行记录
        :return:
        """
        save_dir_log = './prompt/KGC/' + str(self.model) + '/'
        if not os.path.exists(save_dir_log):
            try:
                os.makedirs(save_dir_log)
            except OSError as e:
                print(f"Creating dir'{save_dir_log}': {e}")
        else:
            print(f"Dir '{save_dir_log}' existing!")

        save_log = './prompt/KGC/' + str(self.model) + '/' + str(self.dataset) + '_' + str(self.__class__.__name__) \
                   + '_' + str(self.temperature) + '_' + str(self.tokens) + '_' + str(self.threads) + '.json'
        with open(save_log, 'w') as f:
            for dic in self.memories:
                f.write(json.dumps(dic) + '\n')

    def log_memories(self):
        save_log = './prompt/KGC/' + str(self.model) + '/' + str(self.dataset) + '_' + str(self.__class__.__name__) \
                   + '_' + str(self.temperature) + '_' + str(self.tokens) + '_' + str(self.threads) + '.json'
        self.memories = load_ndjson(save_log)

    def get_lat_lng(self, input, index):
        """
        返回 head entity 和 tail entity 的经纬度
        :param input:
        :param index:
        :return:
        """
        head_lat_lng = input[index]['head geometry value']
        tail_lat_lng = input[index]['tail geometry value']

        return head_lat_lng, tail_lat_lng

    def get_geohash(self, head_lat_lng, tail_lat_lng):

        head_geohash_code = geohash_code(head_lat_lng)
        tail_geohash_code = geohash_code(tail_lat_lng)

        return head_geohash_code, tail_geohash_code

    def get_distance(self, head_lat_lng, tail_lat_lng):

        return distance(head_lat_lng, tail_lat_lng)

    def get_point_belong_polygon(self, head_lat_lng, tail_lat_lng):

        return point_belong_polygon(head_lat_lng, tail_lat_lng)

    def get_point_intersects_linestring(self, head_lat_lng, tail_lat_lng):

        return point_intersects_linestring(head_lat_lng, tail_lat_lng)

    def get_linestring_intersect_ploygon(self, head_lat_lng, tail_lat_lng):

        return linestring_intersect_ploygon(head_lat_lng, tail_lat_lng)

    def get_linestring_belong_ploygon(self, head_lat_lng, tail_lat_lng):

        return linestring_belong_ploygon(head_lat_lng, tail_lat_lng)

    def get_polygon_intersect_ploygon(self, head_lat_lng, tail_lat_lng):

        return polygon_intersect_ploygon(head_lat_lng, tail_lat_lng)

    def get_polygon_belong_ploygon(self, head_lat_lng, tail_lat_lng):

        return polygon_belong_ploygon(head_lat_lng, tail_lat_lng)

class EvaluateAgent_base:
    def __init__(self, evaluation_url, evaluation_headers, evaluation_model, temperature, evaluation_tokens, threads, dataset):
        self.evaluation_url = evaluation_url
        self.evaluation_headers = evaluation_headers
        self.evaluation_model = evaluation_model
        self.temperature = temperature
        self.evaluation_tokens = evaluation_tokens
        self.threads = threads
        self.dataset = dataset
        self.memories = []
    @abstractmethod
    def prompt_construction(self, input):
        """
        construct the prompt completion for Agent

        :param input:
        :return:
        """
        pass

    def memory(self, input, index, response):
        input[index]['score'] = response

        # The memory for Entity Agent
        self.memories.append(copy.deepcopy(input)[index])

    def request(self, input, prompt_completion, index):
        flag = False
        while flag is not True:
            try:
                response = requests.post(self.evaluation_url, headers=self.evaluation_headers, data=json.dumps(prompt_completion))
                response_json = response.json()
                results = response_json['choices'][0]['message']['content']

                # 写入 memory
                self.memory(input, index, results)

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

    def communication(self, index):
        """
        返回抽取出来的 entity types

        :param instruction:
        :return:
        """
        return self.memories[index]['response']
