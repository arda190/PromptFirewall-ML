

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

        return true_positive,true_negative,false_positive,false_negative





    def accuracy(self, y_test,predictions):
        total_test = len(y_test)
        correct = 0

        for i in range(total_test):
            if y_test[i] == predictions[i]:
               correct += 1

        return correct / total_test

    def precision(self, y_test,predictions):
        total_positive = 0
        true_positive = 0

        for i in range(len(y_test)):
            if predictions[i] == 1:
                total_positive += 1
                if y_test[i] == 1:
                    true_positive += 1

        if total_positive == 0:
            return 0

        return true_positive / total_positive


    def recall(self, y_test,predictions):

        total_positive = 0
        true_positive = 0
        false_negative = 0

        for i in range(len(y_test)):
            if y_test[i] == 1:
               total_positive += 1

               if predictions[i] == 1:
                  true_positive += 1
               else:
                  false_negative += 1

        if total_positive == 0:
            return 0

        return true_positive / total_positive


    def f1_score(self, y_test,predictions):
        precision = self.precision(y_test, predictions)
        recall = self.recall(y_test, predictions)

        if precision + recall == 0:
            return 0

        return 2 * precision * recall / (precision + recall)
