from models.base import TEAgent_base
import threading
import requests
import json
from tqdm import tqdm
import copy

class Agent_entity(TEAgent_base):

    def __init__(self, args):

        super(Agent_entity, self).__init__(args.url, args.headers, args.model, args.temperature, args.tokens, args.threads, args.dataset)
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
                {"role": "user",
                 "content": "Given sentence:." + str(text)},
                {"role": "user",
                 "content": "What types of entities are included in this sentence? Return the results with the ['Entity type 1, Entity type 2, ...'] format}"},
            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }

        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)

        return prompt_completion
