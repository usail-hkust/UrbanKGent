import copy
from shapely.geometry import Point, LineString, Polygon
from geopy.geocoders import Nominatim
import geohash2
import os
from models.base import EvaluateAgent_base

class EvaluatorAgent(EvaluateAgent_base):

    def __init__(self, args):

        super(EvaluatorAgent, self).__init__(args.evaluation_url, args.evaluation_headers, args.evaluation_model, args.temperature, args.evaluation_tokens, args.threads, args.dataset)
        self.evaluation_url = self.evaluation_url
        self.evaluation_headers = self.evaluation_headers
        self.evaluation_model = self.evaluation_model
        self.temperature = self.temperature
        self.evaluation_tokens = self.evaluation_tokens
        self.threads = self.threads
        self.dataset = self.dataset
        self.memories = []


class Agent_evaluation_TE(EvaluatorAgent):

    def prompt_construction(self, input, index):

        selected_keys = ["entity name", "entity type", "text description", "triplet"]
        results = {key: input[index].get(key) for key in selected_keys}

        prompt_completion = {
            "model": self.evaluation_model,
            "messages": [
                {"role": "user",
                 "content": "Given the entity name, entity type, text description of entity and the extracted triplets from these informaton. "
                            "Please evaluate the quality of extracted triplets and output the number of true and false triplets and give your confidence value (from 0 to 5)." + '\n'},
                {"role": "user", "content": str(results) + '\n'},
                {"role": "user", "content": "Return the results with the following format without any other explanation: {\"Number of true triplet\": \"1\", \"Number of false triplet\": \"1\", \"Confidence\": \"5\"}."},
            ],
            "temperature": self.temperature,
            "tokens": self.evaluation_tokens,
        }

        return prompt_completion


class Agent_evaluation_KGC(EvaluatorAgent):

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

        head_geohash_code = self.get_geohash_code(head_lat_lng)
        tail_geohash_code = self.get_geohash_code(tail_lat_lng)

        return head_geohash_code, tail_geohash_code


    def get_geohash_code(self, input_geometry):

        if "POINT" in input_geometry:
            # Process Point type
            point_coords = input_geometry.replace("POINT", "").strip("() ")
            longitude, latitude = map(float, point_coords.split())
        elif "LINESTRING" in input_geometry:
            # Process Linestring type
            linestring_coords = input_geometry.replace("LINESTRING", "").strip("() ")
            coordinates = [map(float, coord.split()) for coord in linestring_coords.split(',')]
            longitude, latitude = coordinates[0]
        elif "POLYGON" in input_geometry:
            # Process Polygon type
            polygon_coords = input_geometry.replace("POLYGON", "").strip("() ")
            coordinates = [map(float, coord.split()) for coord in polygon_coords.split(',')]
            longitude, latitude = coordinates[0]
        else:
            raise ValueError("Unsupported geometry type")


        # Encode geohash based on the location
        geohash_code = geohash2.encode(latitude, longitude, precision=8)

        return geohash_code

    def prompt_construction(self, input, index):

        head_lat_lng, tail_lat_lng = self.get_lat_lng(input, index)
        head_geohash_code, tail_geohash_code = self.get_geohash(head_lat_lng, tail_lat_lng)
        RCC8 = input[index]['Geo relation']

        prompt_completion = {
            "model": self.evaluation_model,
            "messages": [
                {"role": "user",
                 "content": "Given the geohash code, latitude and longitude of two geospatial entities, please determine whether the RCC relation between them is correct. RCC consists of 5 basic relations that are possible between two geospatial entities: (1) Disconnected (DC); (2) Externally connected (EC); (3) Equal (EQ); (4) Partially Overlapping (PO); (5) Tangential and nontangential proper parts (IN)." + '\n'},
                {"role": "user",
                 "content": "Entity 1. Geohash: " + head_geohash_code + ". Latitude and Longitude: " + head_lat_lng},
                {"role": "user",
                 "content": "Entity 2. Geohash: " + tail_geohash_code + ". Latitude and Longitude: " + tail_lat_lng},
                {"role": "user", "content": '\n' + "RCC relation: " + str(RCC8) + "\n"},
                {"role": "user", "content": "Output the your judement (True or False) and your confidence (from 0 to 5)."
                                            "Return the results with the following format without any other explanation: {\"Result\": \"True\", \"Confidence\": \"2\"}"},
            ],
            "temperature": self.temperature,
            "tokens": self.evaluation_tokens,
        }

        return prompt_completion

