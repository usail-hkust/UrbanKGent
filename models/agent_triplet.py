from models.base import TEAgent_base
import threading
import requests
import json
from tqdm import tqdm
import copy

class Agent_triplet_COT(TEAgent_base):

    def __init__(self, args, demonstration_number):

        super(Agent_triplet_COT, self).__init__(args.url, args.headers, args.model, args.temperature, args.tokens, args.threads, args.dataset)
        self.url = self.url
        self.headers = self.headers
        self.model = self.model
        self.temperature = self.temperature
        self.tokens = self.tokens
        self.threads = self.threads
        self.dataset = self.dataset
        self.demonstration_number = demonstration_number
        self.memories = []

class Agent_ablation_triplet_COT_demonstration(Agent_triplet_COT):

    def prompt_construction(self, input, index, ent, rel):

        text = input[index]['text description']

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user",
                     "content": "Suppose you are an urban triplet extraction model and I will provide you some urban text. Please recognize the entities with types: " +
                                ent + ", and then recognize the relationships with types: " + rel + " between them." +
                                "Return the result with the {'head entity': ' ', 'relation': ' ', 'tail entity': ' '} format."},
                    {"role": "user", "content": "Given the urban text: " + text},
                    {"role": "user", "content": "Let's think step by step."}
                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion



class Agent_spatial_triplet_COT_demonstration(Agent_triplet_COT):

    def prompt_construction_wo_ent_rel(self, input, index):
        text = input[index]['text description']

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "Suppose you are an urban spatial triplet extraction model and I will provide you some text data. Please recognize the entities and then recognize spatial relations between them." +
                            "Return the result with the {'head entity': ' ', 'relation': ' ', 'tail entity': ' '} format."},
                {"role": "user", "content": "Given the urban text: " + text + '\n'},
                {"role": "user", "content": "Let's think step by step."}
            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)

        return prompt_completion


    def prompt_construction(self, input, index, ent, rel):

        text = input[index]['text description']

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user",
                     "content": "Suppose you are an urban spatial triplet extraction model and I will provide you some text data. Please recognize the entities with types: " +
                                ent + ", and then recognize spatial relations types: " + rel + " between them." +
                                "Return the result with the {'head entity': ' ', 'relation': ' ', 'tail entity': ' '} format."},
                {"role": "user", "content": "Given the urban text: " + text + '\n'},
                    {"role": "user", "content": "Let's think step by step."}
                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)

        return prompt_completion

class Agent_temporal_triplet_COT_demonstration(Agent_triplet_COT):

    def prompt_construction_wo_ent_rel(self, input, index):
        text = input[index]['text description']

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user",
                     "content": "Suppose you are an urban temporal triplet extraction model and I will provide you some text data. Please recognize the entities, and then recognize temporal relations types between them." +
                                "Return the result with the {'head entity': ' ', 'relation': ' ', 'tail entity': ' '} format."},
                {"role": "user", "content": "Given the urban text: " + text + '\n'},
                    {"role": "user", "content": "Let's think step by step."}
                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion

    def prompt_construction(self, input, index, ent, rel):

        text = input[index]['text description']

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user",
                     "content": "Suppose you are an urban temporal triplet extraction model and I will provide you some text data. Please recognize the entities with types: " +
                                ent + ", and then recognize temporal relations types: " + rel + " between them." +
                                "Return the result with the {'head entity': ' ', 'relation': ' ', 'tail entity': ' '} format."},
                {"role": "user", "content": "Given the urban text: " + text + '\n'},
                {"role": "user", "content": "Let's think step by step."}
                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion

class Agent_funtional_triplet_COT_demonstration(Agent_triplet_COT):

    def prompt_construction_wo_ent_rel(self, input, index):
        text = input[index]['text description']

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user",
                     "content": "Suppose you are an urban functional triplet extraction model and I will provide you some text data. Please recognize the entities, and then recognize functional relations between them." +
                                "Return the result with the {'head entity': ' ', 'relation': ' ', 'tail entity': ' '} format."},
                {"role": "user", "content": "Given the urban text: " + text + '\n'},
                    {"role": "user", "content": "Let's think step by step."}
                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion


    def prompt_construction(self, input, index, ent, rel):

        text = input[index]['text description']

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user",
                     "content": "Suppose you are a urban functional triplet extraction model and I will provide you some text data. Please recognize the entities with types: " +
                                ent + ", and then recognize functional relations types: " + rel + " between them." +
                                "Return the result with the {'head entity': ' ', 'relation': ' ', 'tail entity': ' '} format."},
                {"role": "user", "content": "Given the urban text: " + text + '\n'},
                    {"role": "user", "content": "Let's think step by step."}
                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion


class Agent_triplet_ICL(TEAgent_base):

    def __init__(self, args):

        super(Agent_triplet_ICL, self).__init__(args.url, args.headers, args.model, args.temperature, args.tokens, args.threads, args.dataset)
        self.url = self.url
        self.headers = self.headers
        self.model = self.model
        self.temperature = self.temperature
        self.tokens = self.tokens
        self.dataset = self.dataset
        self.memories = []

class Agent_ablation_triplet_ICL(Agent_triplet_ICL):

    def prompt_construction(self, input, index, demonstration):

        text = input[index]['text description']

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": demonstration + '\n'},
                    {"role": "user", "content": "Extract triplet from the following urban text:" + text + '\n'},
                    {"role": "user", "content": "Let's think step by step."}
                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion


class Agent_spatial_triplet_ICL(Agent_triplet_ICL):

    def prompt_construction(self, input, index, demonstration):

        text = input[index]['text description']

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": demonstration + '\n'},
                    {"role": "user", "content": "Extract triplet related to spatial relationship from the following urban text:" + text + '\n'},
                    {"role": "user", "content": "Let's think step by step."}
                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion


class Agent_temporal_triplet_ICL(Agent_triplet_ICL):

    def prompt_construction(self, input, index, demonstration):

        text = input[index]['text description']

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": demonstration + '\n'},
                    {"role": "user", "content": "Extract triplet related to temporal relationship from the following urban text:" + text + '\n'},
                    {"role": "user", "content": "Let's think step by step."}
                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion


class Agent_functional_triplet_ICL(Agent_triplet_ICL):

    def prompt_construction(self, input, index, demonstration):

        text = input[index]['text description']

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": demonstration + '\n'},
                    {"role": "user", "content": "Extract triplet related to functional relationship from the following urban text:" + text + '\n'},
                    {"role": "user", "content": "Let's think step by step."}
                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion
