import json
from utils.load_json import *
import csv
import re

def KGC_acc_confidence(path):

    data = load_ndjson(path)
    results = []
    confidence = []
    for index in range(len(data)):
        pattern__ = r"{\"Result\"(.*?)\"}"
        matche_ = re.search(pattern__, data[index]['score'], re.IGNORECASE | re.DOTALL)
        results_json = json.loads(matche_[0])

        results_index = results_json['Result']
        confidence_index = int(results_json['Confidence'])

        results.append(results_index)
        confidence.append(confidence_index)

    accuracy = sum(1 for pred in results if pred == 'True') / len(results)
    avg_confidence = sum(confidence) / len(confidence)

    with open(path + '_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Metric', 'Value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'Metric': 'Accuracy', 'Value': f'{accuracy:.2%}'})
        writer.writerow({'Metric': 'Average Confidence', 'Value': f'{avg_confidence:.2f}'})

    return accuracy, avg_confidence


def TE_acc_confidence(path):
    data = load_ndjson(path)

    results_true = []
    results_false = []
    confidence = []
    for index in range(len(data)):

        pattern__ = r"{\"Number of true triplet\"(.*?)\"}"
        matche_ = re.search(pattern__, data[index]['score'], re.IGNORECASE | re.DOTALL)
        results_json = json.loads(matche_[0])

        results_true_index = int(results_json['Number of true triplet'])
        results_false_index = int(results_json['Number of false triplet'])
        confidence_index = int(results_json['Confidence'])

        results_true.append(results_true_index)
        results_false.append(results_false_index)
        confidence.append(confidence_index)

    total_true = sum(results_true)
    total_false = sum(results_false)
    total_samples = total_true + total_false

    accuracy = total_true / total_samples
    avg_confidence = sum(confidence) / len(confidence)

    with open(path + '_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Metric', 'Value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'Metric': 'Accuracy', 'Value': f'{accuracy:.2%}'})
        writer.writerow({'Metric': 'Average Confidence', 'Value': f'{avg_confidence:.2f}'})

    return accuracy, avg_confidence