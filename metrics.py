from torchmetrics import (
    F1Score,
    PrecisionRecallCurve,
    ROC,
    Accuracy,
    AUROC,
    ConfusionMatrix,
)
from torchmetrics.classification import BinaryStatScores
import matplotlib.pyplot as plt
import os
from time import sleep

class Metrics:
    def __init__(self, probs, labels,two_d_probs, fig_prefix, wandb, classes):
        self.probs = probs.to("cpu")
        self.labels = labels.to("cpu")
        self.two_d_probs = two_d_probs.to("cpu")
        self.fig_prefix = fig_prefix
        self.fig_dir = "figures/"+self.fig_prefix
        os.makedirs(self.fig_dir, exist_ok = True)
        self.wandb = wandb
        self.classes = classes
        
    def get_stats(self, save_plot=True):
        accuracy = self.calc_accuracy()
        auroc = self.calc_auroc()
        confusion_metrics = self.extract_confusion_matrix()
        f1 = self.calc_f1_score()
        if save_plot:
            self.calc_pr_curve()
            self.calc_ROC_curve()
            self.calc_sensitivity_specificity_curve()
        return (accuracy ,auroc ,confusion_metrics ,f1)
            
    def calc_accuracy(self):
        accuracy = Accuracy(task="binary")
        return accuracy(self.probs, self.labels).item()

    def calc_auroc(self):
        auroc = AUROC(task="binary")
        return auroc(self.probs, self.labels).item()

    def extract_confusion_matrix(self):
        confmat = ConfusionMatrix(task="binary", num_classes=2)
        return confmat(self.probs, self.labels)

    def calc_f1_score(self):
        f1 = F1Score(task="binary")
        return f1(self.probs, self.labels).item()

    def calc_pr_curve(self):
        pr_curve = PrecisionRecallCurve(task="binary")
        precision, recall, thresholds = pr_curve(self.probs, self.labels)
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall (PR) Curve")
        plt.savefig(f'{self.fig_dir}/PR_curve.png')
        plt.close()
        sleep(1)
        if self.wandb:
            self.wandb.log({"pr": self.wandb.plot.pr_curve(self.labels, self.two_d_probs, self.classes)})
            self.wandb.log({"pr_curve": self.wandb.Image(f'{self.fig_dir}/PR_curve.png')})

    def calc_ROC_curve(self):
        roc = ROC(task="binary")
        fpr, tpr, thresholds = roc(self.probs, self.labels)
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.savefig(f'{self.fig_dir}/ROC_curve.png')
        plt.close()
        sleep(1)
        if self.wandb:
            self.wandb.log({"roc": self.wandb.plot.roc_curve(self.labels, self.two_d_probs, self.classes)})
            self.wandb.log({"roc_curve": self.wandb.Image(f'{self.fig_dir}/ROC_curve.png')})
        

    def calc_sensitivity_specificity_curve(self):
        num = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        sens = []
        spec = []
        for i in num:
            metric = BinaryStatScores(threshold=i)
            tp, fp, tn, fn, support = metric(self.probs, self.labels)
            Sensitivity = round(tp.item() / (tp.item() + fn.item()), 2)
            Specificity = round(tn.item() / (tn.item() + fp.item()), 2)
            sens.append(Sensitivity)
            spec.append(Specificity)

        plt.plot(num, sens, label="Sensitivity")
        plt.plot(num, spec, label="Specificity")
        plt.xlabel("Threshold")
        plt.ylabel("Value")
        plt.title("Sensitivity and Specificity vs. Threshold")
        plt.legend()
        plt.savefig(f'{self.fig_dir}/SensSpecCurve.png')
        plt.close()
        sleep(1)
        if self.wandb:
            self.wandb.log({"my_custom_id" : self.wandb.plot.line_series(
            xs=num,
            ys=[sens, spec],
            keys=["Sensitivity", "Specificity"],
            title="Sensitivity and Specificity vs. Threshold",
            xname="Threshold")})
            self.wandb.log({"SensSpecCurve": self.wandb.Image(f'{self.fig_dir}/SensSpecCurve.png')})

