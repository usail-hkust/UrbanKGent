from models.base import TEAgent_base
import threading
import requests
import json
from tqdm import tqdm
import copy

class Agent_relation(TEAgent_base):

    def __init__(self, args):

        super(Agent_relation, self).__init__(args.url, args.headers, args.model, args.temperature, args.tokens, args.threads, args.dataset)
        self.url = self.url
        self.headers = self.headers
        self.model = self.model
        self.temperature = self.temperature
        self.tokens = self.tokens
        self.threads = self.threads
        self.dataset = self.dataset
        self.memories = []

class Agent_ablation_relation(Agent_relation):

    def prompt_construction(self, input, index):

        text = input[index]['text description']

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "Given sentence:." + str(text)},
                {"role": "user",
                 "content": "What types of relations are included in this sentence? Return the results with the ['relation type 1, relation type 2, ...'] format}"},
            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion


class Agent_spatial_relation(Agent_relation):

    def prompt_construction(self, input, index):

        text = input[index]['text description']

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "Given sentence:." + str(text)},
                {"role": "user",
                 "content": "Spatial relation specifies how some object is located in space in relation to some reference object. What types of spatial relations are included in this sentence? Return the results with the ['Spatial relation type 1, Spatial relation type 2, ...'] format}"},
            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion


class Agent_temporal_relation(Agent_relation):

    def prompt_construction(self, input, index):

        text = input[index]['text description']

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "Given sentence:." + str(text)},
                {"role": "user",
                 "content": "Temporal relation communicates the simultaneity or ordering in time of events or states. What types of temporal relations are included in this sentence? Return the results with the ['Temporal relation type 1, Temporal relation type 2, ...'] format}"},
            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion


class Agent_functional_relation(Agent_relation):

    def prompt_construction(self, input, index):

        text = input[index]['text description']

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "Given sentence:." + str(text)},
                {"role": "user",
                 "content": "Functional relation specifies the current term is a type of urban entity. What types of functional relations are included in this sentence? Return the results with the ['Functional relation type 1, Functional relation type 2, ...'] format}"},
            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion
