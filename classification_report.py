
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
class ClassificationReport():
    def __init__(self, predicted_labels, true_labels):
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
        avg = ClassificationReport([], [])
        for r in report_list:
            avg.accuracy += r.accuracy
            avg.cohen_kappa += r.cohen_kappa
            avg.accuracy += r.accuracy
        pass