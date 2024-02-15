from models.base import KGCAgent_base
import threading
import requests
import json
from tqdm import tqdm
import copy

class Vanilla_KGC(KGCAgent_base):

    def __init__(self, args):

        super(Vanilla_KGC, self).__init__(args.url, args.headers, args.model, args.temperature, args.tokens, args.threads, args.dataset)
        self.url = self.url
        self.headers = self.headers
        self.model = self.model
        self.temperature = self.temperature
        self.tokens = self.tokens
        self.threads = self.threads
        self.dataset = self.dataset
        self.memories = []


    def prompt_construction(self, input, index):

        head_lat_lng, tail_lat_lng = self.get_lat_lng(input, index)

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "Given the latitude and longitude of two geospatial entities, please use the region connection calculus (RCC) to describes the geospatial relationships between the two geospatial entities."},
                {"role": "user",
                 "content": "RCC8 consists of 8 basic relations that are possible between two geospatial entities: disconnected (DC), externally connected (EC), equal (EQ), partially overlapping (PO), tangential proper part (TPP), tangential proper part inverse (TPPi), non-tangential proper part (NTPP) and non-tangential proper part inverse (NTPPi)." + '\n'},
                {"role": "user", "content": "Following the above definition, output the geospatial relation between the two geospatial entities: "},
                {"role": "user", "content": "Entity 1. Latitude and Longitude: " + head_lat_lng},
                {"role": "user", "content": "Entity 2. Latitude and Longitude: " + tail_lat_lng + '\n'},
                {"role": "user", "content": "Let's think step by step"}

            ],
            "temperature": self.temperature,
            "tokens": self.tokens,

        }

        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)

        return prompt_completion