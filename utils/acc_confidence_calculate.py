import json
from utils.load_json import *
import csv

def KGC_acc_confidence(path):

    data = load_ndjson(path)
    results = []
    confidence = []
    for index in range(len(data)):

        try:
            temp_index = eval(data[index]['score'])
            results_index = temp_index['Result']
            confidence_index = int(temp_index['Confidence'])
        except:
            results_index = False
            confidence_index = 1
        results.append(results_index)
        confidence.append(confidence_index)

    accuracy = sum(results) / len(results)
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

        try:
            temp_index = eval(data[index]['score'])
            results_true_index = int(temp_index['Number of true triplet'])
            results_false_index = int(temp_index['Number of false triplet'])
            confidence_index = int(temp_index['Confidence'])
        except:
            results_true_index = 0
            results_false_index = 0
            confidence_index = 1

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