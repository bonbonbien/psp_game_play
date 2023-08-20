from sklearn.metrics import f1_score
from torchmetrics import AUROC
from skopt import gp_minimize

class Evaluator(object):
    """Evaluator.

    Parameters:
        metric_names: evaluation metrics
        n_qns: number of questions
    """

    def __init__(self, metric_names, n_qns):
        self.metric_names = metric_names
        self.n_qns = n_qns
        self.eval_metrics = {}
        self._build()

    def evaluate(self, y_true, y_pred):
        """Run evaluation using pre-specified metrics.

        Parameters:
            y_true: groundtruths
            y_pred: predicting results

        Return:
            eval_result: evaluation performance report
        """
        eval_result = {}
        for metric_name, metric in self.eval_metrics.items():
            eval_result[metric_name] = metric(y_pred, y_true)
        return eval_result

    def _build(self) -> None:
        """Build evaluation metric instances."""
        for metric_name in self.metric_names:
            if metric_name == "auroc":
                self.eval_metrics[metric_name] = self._AUROC
            elif metric_name == "f1":
                self.eval_metrics[metric_name] = self.f1_macro

    def _AUROC(self, y_pred, y_true):
        """Area Under the Receiver Operating Characteristic curve.

        Parameters:
            y_pred: predicting results
            y_true: groundtruths

        Return:
            auroc: area under the receiver operating characteristic
                curve
        """
        metric = AUROC(task="multilabel", num_labels=self.n_qns)
        _ = metric(y_pred, y_true.int())
        auroc = metric.compute().item()
        metric.reset()

        return auroc
    
    def get_best_threshold_and_score(self, ts, ps, start=0.2, end=0.8, step=0.01, n_calls=100):
        def f(t):
            _ps = (ps > t).astype("int")
            return -f1_score(ts, _ps, average="macro")

        res = gp_minimize(
            f,  # the function to minimize
            [(0.2, 0.8)],  # the bounds on each dimension of x
            acq_func="EI",  # the acquisition function
            n_calls=n_calls,  # the number of evaluations of f
            n_random_starts=3,  # the number of random initialization points
            random_state=42,
        )  # the random seed

        bt = res.x[0]
        s = res.fun
        ts = [xs[0] for xs in res.x_iters]
        ss = [-xs for xs in res.func_vals]
        return bt, -s, ts, ss

    def f1_macro(self, pred, true):
        y_pred = pred.cpu().numpy().reshape(-1)
        y_true = true.cpu().numpy().reshape(-1)
        _, s, _, _ = self.get_best_threshold_and_score(y_true, y_pred, n_calls=20,)
        return s
        