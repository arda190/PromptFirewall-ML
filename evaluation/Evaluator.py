

class Evaluator:
    def __init__(self):
        pass


    def confusion_matrix(self, y_test,predictions):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for i in range(len(y_test)):
            if y_test[i] == 1:
                if predictions[i] == 1:
                    true_positive += 1
                else:
                    false_negative += 1

            else:
                if predictions[i] == 1:
                    false_positive += 1
                else:
                    true_negative += 1

        return {
            "TP": true_positive,
            "TN": true_negative,
            "FP": false_positive,
            "FN": false_negative
        }





    def accuracy(self, cm):

        tp = cm["TP"]
        tn = cm["TN"]
        fp = cm["FP"]
        fn = cm["FN"]

        return (tp + tn) / (tp + tn + fp + fn)

    def precision(self, cm):

        tp = cm["TP"]
        fp = cm["FP"]

        if tp+fp == 0:
            return 0

        return  tp / (tp + fp)


    def recall(self, cm):

        tp = cm["TP"]
        fn = cm["FN"]

        if tp+fn == 0:
            return 0

        return tp / (tp + fn)


    def f1_score(self,cm):
        precision = self.precision(cm)
        recall = self.recall(cm)

        if precision + recall == 0:
            return 0

        return 2 * precision * recall / (precision + recall)



    def evaluate(self, y_test, predictions):

        cm = self.confusion_matrix(y_test, predictions)

        return {
            "accuracy": self.accuracy(cm),
            "precision": self.precision(cm),
            "recall": self.recall(cm),
            "f1_score": self.f1_score(cm),
            "confusion_matrix": cm
        }