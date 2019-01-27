
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
import numpy as np

class ClassificationReport():
    def __init__(self, predicted_labels, true_labels):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.accuracy = accuracy_score(true_labels, predicted_labels)
        self.cohen_kappa = cohen_kappa_score(true_labels, predicted_labels)
        self.confusion_matrix = confusion_matrix(true_labels, predicted_labels)
        self.report = classification_report(true_labels, predicted_labels)

    def __str__(self):
        return str(self.confusion_matrix) + "\n" \
               + str(self.report) + "\n" \
               + "Cohen Kappa : {}".format(self.cohen_kappa) + "\n" \
               + "Accuracy : {}".format(self.accuracy)
    @staticmethod
    def average(report_list):
        all_true = []
        all_predicted = []
        for r in report_list:
            all_true.extend(r.true_labels)
            all_predicted.extend(r.predicted_labels)
        return ClassificationReport(all_predicted, all_true)