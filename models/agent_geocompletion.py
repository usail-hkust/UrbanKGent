from models.base import KGCAgent_base
import threading
import requests
import json
from tqdm import tqdm
import copy

class GEOKGC_Agent_base(KGCAgent_base):

    def __init__(self, args):

        super(GEOKGC_Agent_base, self).__init__(args.url, args.headers, args.model, args.temperature, args.tokens, args.threads, args.dataset)
        self.url = self.url
        self.headers = self.headers
        self.model = self.model
        self.temperature = self.temperature
        self.tokens = self.tokens
        self.threads = self.threads
        self.dataset = self.dataset
        self.memories = []

class GEO_RCC8_COT_Agent(GEOKGC_Agent_base):

    def prompt_construction(self, input, index):

        head_lat_lng, tail_lat_lng = self.get_lat_lng(input, index)

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "Given two geospatial entities, please use the region connection calculus (RCC) to describes the geospatial relationships between the two geospatial entities."},
                {"role": "user",
                 "content": "RCC8 consists of 8 basic relations that are possible between two geospatial entities: disconnected (DC), externally connected (EC), equal (EQ), partially overlapping (PO), tangential proper part (TPP), tangential proper part inverse (TPPi), non-tangential proper part (NTPP) and non-tangential proper part inverse (NTPPi)." + '\n'},
                {"role": "user", "content": "Following the above definition, output the geospatial relation between the two geospatial entities: "},
                {"role": "user", "content": "Entity 1: Latitude and Longitude: " + head_lat_lng},
                {"role": "user", "content": "Entity 2: Latitude and Longitude: " + tail_lat_lng + '\n'},
                {"role": "user", "content": "Let's think step by step"}

            ],
            "temperature": self.temperature,
            "tokens": self.tokens,

        }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion


class GEO_RCC8_Tool_Agent(GEOKGC_Agent_base):

    def prompt_construction(self, input, index, COT):

        head_lat_lng, tail_lat_lng = self.get_lat_lng(input, index)

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "Given two geospatial entities, please use the region connection calculus (RCC) to describes the geospatial relationships between the two geospatial entities."},
                {"role": "user",
                 "content": "RCC8 consists of 8 basic relations that are possible between two geospatial entities: disconnected (DC), externally connected (EC), equal (EQ), partially overlapping (PO), tangential proper part (TPP), tangential proper part inverse (TPPi), non-tangential proper part (NTPP) and non-tangential proper part inverse (NTPPi)."+ '\n'},
                {"role": "user",
                 "content": "Following the above definition, output the geospatial relation between the two geospatial entities: "},
                {"role": "user", "content": "Entity 1. Latitude and Longitude: " + head_lat_lng},
                {"role": "user", "content": "Entity 2. Latitude and Longitude: " + tail_lat_lng + '\n'},
                {"role": "user", "content": "Let's think step by step." + COT + '\n'},
                {"role": "user", "content": "The above reasoning process is not completly right. To further promise the results are correct, we can call some external tool interface to help analysis." + '\n'},
                {"role": "user", "content": "I can provide 8 types of tool interface, and their function are as follows: "
                                            "(1) geohash encoding; (2) distance calculation; (3) identify if a point belongs to a polygon; (4) identify if a point intersects a linestring; (5) identify if a linestring intersects to a polygon; (6) identify if a linestring belongs to a polygon; (7) identify if a polygon intersects to a polygon; (8) identify if a polygon belongs to a polygon." + '\n'},
                {"role": "user", "content": "Tell me which types of tool interface you may need to better finish reasoning. Output the name and number of tool interface without other textual description."},
            ],
            "temperature": self.temperature,
            "tokens": self.tokens,

        }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion

class GEO_RCC8_Tool_COT_Agent(GEOKGC_Agent_base):

    def prompt_construction(self, input, index, COT,tool):

        head_lat_lng, tail_lat_lng = self.get_lat_lng(input, index)

        head_geoash = ''
        tail_geoash = ''
        distance = ''
        belong_to = ''
        intersects = ''
        if 'geohash' in tool or '1' in tool:
            head_geoash, tail_geoash = self.get_geohash(head_lat_lng, tail_lat_lng)

            head_geoash = str(head_geoash)
            tail_geoash = str(tail_geoash)
        if 'distance' in tool or '2' in tool:
            distance = str(self.get_distance(head_lat_lng, tail_lat_lng))
        if "point belongs to a polygon" in tool or '3' in tool:
            if input[index]['head geometry type'] == 'coordinate' and input[index]['tail geometry type'] == 'polygon':
                belong_to = str(self.get_point_belong_polygon(head_lat_lng, tail_lat_lng))
        if "point intersects a linestring" in tool or '4' in tool:
            if input[index]['head geometry type'] == 'coordinate' and input[index]['tail geometry type'] == 'linestring':
                intersects = str(self.get_point_intersects_linestring(head_lat_lng, tail_lat_lng))
        if "linestring intersects a polygon" in tool or '5' in tool:
            if input[index]['head geometry type'] == 'linestring' and input[index]['tail geometry type'] == 'polygon':
                intersects = str(self.get_linestring_intersect_ploygon(head_lat_lng, tail_lat_lng))
        if "linestring belongs to a polygon" in tool or '6' in tool:
            if input[index]['head geometry type'] == 'linestring' and input[index]['tail geometry type'] == 'polygon':
                belong_to = str(self.get_linestring_belong_ploygon(head_lat_lng, tail_lat_lng))
        if "polygon intersects a polygon" in tool or '7' in tool:
            if input[index]['head geometry type'] == 'polygon' and input[index]['tail geometry type'] == 'polygon':
                intersects = str(self.get_polygon_intersect_ploygon(head_lat_lng, tail_lat_lng))
        if "polygon belongs to a polygon" in tool or '8' in tool:
            if input[index]['head geometry type'] == 'polygon' and input[index]['tail geometry type'] == 'polygon':
                belong_to = str(self.get_polygon_belong_ploygon(head_lat_lng, tail_lat_lng))

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "Given two geospatial entities, please use the region connection calculus (RCC) to describes the geospatial relationships between the two geospatial entities."},
                {"role": "user",
                 "content": "RCC8 consists of 8 basic relations that are possible between two geospatial entities: disconnected (DC), externally connected (EC), equal (EQ), partially overlapping (PO), tangential proper part (TPP), tangential proper part inverse (TPPi), non-tangential proper part (NTPP) and non-tangential proper part inverse (NTPPi)." + '\n'},
                {"role": "user",
                 "content": "Following the above definition, output the geospatial relation between the two geospatial entities: "},
                {"role": "user", "content": "Entity 1. Latitude and Longitude: " + head_lat_lng},
                {"role": "user", "content": "Entity 2. Latitude and Longitude: " + tail_lat_lng + '\n'},
                {"role": "user", "content": "Let's think step by step." + COT + '\n'},
                {"role": "user", "content": "By calling some tool interface, we now have updated information as follows:" + '\n'},
                {"role": "user", "content": "The geohash of Entity 1 is:" + head_geoash + ". And the geohash of Entity 2 is:" + tail_geoash +
                                ". The distance between entity 1 and entity 2 is: " + distance +
                                "km. Moreover, the assertion that 'entity 1 and entity 2 intersect with each other' is: " + intersects +
                                ". The assertion that 'entity 1 is geospatially located within the of entity 2' is " + belong_to + '\n'
                 },
                {"role": "user", "content": "Please refine your reasoning process and output the final answer."},


            ],
            "temperature": self.temperature,
            "tokens": self.tokens,

        }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion

class GEO_RCC8_ICL_Format_Agent(GEOKGC_Agent_base):

    def prompt_construction(self, reasoning_process):

        prompt_completion = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": "RCC8 consists of 8 basic relations that are possible between two geospatial entities: disconnected (DC), externally connected (EC), equal (EQ), partially overlapping (PO), tangential proper part (TPP), tangential proper part inverse (TPPi), non-tangential proper part (NTPP) and non-tangential proper part inverse (NTPPi)." + '\n'},
                {"role": "user",
                 "content": "Here are the reasoning process when using region connection calculus (RCC) to describes the geospatial relationships between the two geospatial entities."
                            "Based on the above reasoning process, please directly output which RCC8 relationship these two geospatial entities belong to. Output the answer without any other textual description." + '\n'},
                {"role": "user", "content": "Reasoning process:" + reasoning_process + "\n"},
                {"role": "user", "content": "Again! Output the answer without any other textual description. Example: Disconnected (DC)."},

            ],
            "temperature": self.temperature,
            "tokens": self.tokens,
        }
        if "llama" in self.model:
            prompt_completion = self.prompt_gpt_to_llama(prompt_completion)
        return prompt_completion
