from models.base import TEAgent_base
import threading
import requests
import json
from tqdm import tqdm
import copy

class Vanilla_TE(TEAgent_base):

    def __init__(self, args):

        super(Vanilla_TE, self).__init__(args.url, args.headers, args.model, args.temperature, args.tokens, args.threads, args.dataset)
        self.url = self.url
        self.headers = self.headers
        self.model = self.model
        self.temperature = self.temperature
        self.tokens = self.tokens
        self.threads = self.threads
        self.dataset = self.dataset
        self.memories = []


    def prompt_construction(self, input, index):

        text = input[index]['text description']

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": "Given sentence: " + str(text) + '\n'},
                {"role": "user",
                 "content": "Suppose you are a knowledge graph construction expert. What triplets are included in this sentence? Return the results in the following format: {'head entity': ' ', 'relation': ' ', 'tail entity': ' '}." + '\n'},
                {"role": "user", "content": "Let's think step by step."},
            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }

        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)

        return prompt_completion


class Agent_triplet_from_Vanilla_TE(TEAgent_base):
    def __init__(self, args):

        super(Agent_triplet_from_Vanilla_TE, self).__init__(args.url, args.headers, args.model, args.temperature, args.tokens, args.threads, args.dataset)
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
                    {"role": "user", "content": "Here are the reasoning process and extracted triplets. Please filter the reasoning process and just return the extracted triplets. Return the result with the {'head entity': ' ', 'relation': ' ', 'tail entity': ' '} format." + '\n'},
                    {"role": "user", "content": "Reasoning process and extracted triplets:" + triplet},

                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion