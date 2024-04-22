from models.base import RTEAgent_base
import threading
import requests
import json
from tqdm import tqdm
import copy

class RTEAgent(RTEAgent_base):

    def __init__(self, args):

        super(RTEAgent, self).__init__(args.url, args.headers, args.model, args.temperature, args.tokens, args.threads, args.dataset)
        self.url = self.url
        self.headers = self.headers
        self.model = self.model
        self.temperature = self.temperature
        self.tokens = self.tokens
        self.threads = self.threads
        self.dataset = self.dataset
        self.memories = []

class RTEAgent_NER_Spatial(RTEAgent):

    def prompt_construction(self, input, index):

        text = input[index]['text description']

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "Given the urban textual sentence: " + str(text) + '\n'},
                {"role": "user",
                 "content": "What types of spatial entities are included in this sentence?"
                            "Spatial specifies how thing occupying some space, entity that can be contained within a region of space."
                            "Return the results with the following format without any other explanation: {\"Spatial entities\": \"[Entity type 1, Entity type 2, ...]\"}"},
            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }

        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)

        return prompt_completion


class RTEAgent_NER_Temporal(RTEAgent):

    def prompt_construction(self, input, index):
        text = input[index]['text description']

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "Given the urban textual sentence: " + str(text) + '\n'},
                {"role": "user",
                 "content": "What types of spatial entities are included in this sentence?"
                            "Temporal entities specifies how thing that can be contained within a period of time, or change in state (e.g. events, periods, acts)."
                            "Return the results with the following format without any other explanation: {\"Temporal entities\": \"[Entity type 1, Entity type 2, ...]\"}"},
            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }

        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)

        return prompt_completion


class RTEAgent_NER_Functional(RTEAgent):

    def prompt_construction(self, input, index):
        text = input[index]['text description']

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "Given the urban textual sentence: " + str(text) + '\n'},
                {"role": "user",
                 "content": "What types of spatial entities are included in this sentence?"
                            "Functional entities specifies any independent party (i.e., person, business entity, governmental entity, or other organization) defined in terms of its function."
                            "Return the results with the following format without any other explanation: {\"Functional entities\": \"[Entity type 1, Entity type 2, ...]\"}"},
            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }

        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)

        return prompt_completion

class RTEAgent_RE_Spatial(RTEAgent):

    def prompt_construction(self, input, index):

        text = input[index]['text description']

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "Given sentence: " + str(text)},
                {"role": "user",
                 "content": "What types of spatial relations are included in this sentence?"
                            "Spatial relation specifies how some object is located in space in relation to some reference object."
                            "Return the results with the following format without any other explanation: {\"Spatial relation\": \"Spatial relation type 1, Spatial relation type 2, ...]\"}"},
            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion


class RTEAgent_RE_Temporal(RTEAgent):

    def prompt_construction(self, input, index):

        text = input[index]['text description']

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "Given sentence:." + str(text)},
                {"role": "user",
                 "content": "What types of temporal relations are included in this sentence?"
                            "Temporal relation communicates the simultaneity or ordering in time of events or states."
                            "Return the results with the following format without any other explanation: {\"Temporal relation\": \"[Temporal relation type 1, Temporal relation type 2, ...]\"}"},
            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion


class RTEAgent_RE_Functional(RTEAgent):

    def prompt_construction(self, input, index):

        text = input[index]['text description']

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "Given sentence:." + str(text)},
                {"role": "user",
                 "content": "What types of functional relations are included in this sentence?"
                            "Functional relation specifies the current term is a type of urban entity."
                            "Return the results with the following format without any other explanation: {\"Functional relation\": \"[ Functional relation type 1, Functional relation type 2, ...]\"}"},
            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion

class RTEAgent_TE_Spatial(RTEAgent):

    def prompt_construction(self, input, index, ent, rel):

        text = input[index]['text description']

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user",
                     "content": "Suppose you are an urban spatial triplet extraction model and I will provide you some text data. Please recognize the spatial entities with types: " +
                                ent + ", and then recognize spatial relations types: " + rel + " between them." + '\n'},
                {"role": "user", "content": "Given the urban text: " + text + '\n'},
                    {"role": "user", "content": "Return the result with the following format: {\"Spatial triplet 1\": {\"Head entity\": \"\", \"Relation\": \"\", \"Tail entity\": \"\"}, \"Spatial triplet 2\": {\"Head entity\": \"\", \"Relation\": \"\", \"Tail entity\": \"\"}}."
                                                "Let's think step by step."
                     }
                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)

        return prompt_completion

class RTEAgent_TE_Temporal(RTEAgent):

    def prompt_construction(self, input, index, ent, rel):

        text = input[index]['text description']

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user",
                     "content": "Suppose you are an urban temporal triplet extraction model and I will provide you some text data. Please recognize the temporal entities with types: " +
                                ent + ", and then recognize temporal relations types: " + rel + " between them." + '\n'},
                {"role": "user", "content": "Given the urban text: " + text + '\n'},
                    {"role": "user", "content": "Return the result with the following format: {\"Temporal triplet 1\": {\"Head entity\": \"\", \"Relation\": \"\", \"Tail entity\": \"\"}, \"Temporal triplet 2\": {\"Head entity\": \"\", \"Relation\": \"\", \"Tail entity\": \"\"}}."
                                                "Let's think step by step."}
                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)

        return prompt_completion

class RTEAgent_TE_Functional(RTEAgent):

    def prompt_construction(self, input, index, ent, rel):

        text = input[index]['text description']

        prompt_completion = {
                "model": self.model,
                "messages": [
                    {"role": "user",
                     "content": "Suppose you are an urban functional triplet extraction model and I will provide you some text data. Please recognize the functional entities with types: " +
                                ent + ", and then recognize spatial relations types: " + rel + " between them." + '\n'},
                {"role": "user", "content": "Given the urban text: " + text + '\n'},
                {"role": "user", "content": "Return the result with the following format: {\"Functional triplet 1\": {\"Head entity\": \"\", \"Relation\": \"\", \"Tail entity\": \"\"}, \"Functional triplet 2\": {\"Head entity\": \"\", \"Relation\": \"\", \"Tail entity\": \"\"}}."
                                            "Let's think step by step."
                     }
                ],
                "temperature": self.temperature,
                "tokens": self.tokens,
            }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)

        return prompt_completion