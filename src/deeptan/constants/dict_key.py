r"""
Dictionary keys and column names.
"""

title_train = "train"
title_val = "val"
title_test = "test"
title_predict = "pred"

abbr_train = "trn"
abbr_val = "val"
abbr_test = "tst"

title_trn_loss = "trn/loss"
title_val_loss = "val/loss"
title_tst_loss = "tst/loss"

tsb_keys2pick = [
    "test/recon_MSE",
    "test/recon_RMSE",
    "test/recon_MAE",
    "test/recon_PCC",
    "test/label_MSE",
    "test/label_RMSE",
    "test/label_MAE",
    "test/label_PCC",
    "test/label_F1_weighted",
    "test/label_F1_macro",
    "test/label_F1_micro",
    "test/label_AUROC",
    "test/label_Accuracy",
    "test/label_Precision",
    "test/label_Recall",
    "test/loss",
    "test/loss_unweighted",
    "val/loss",
    "val/loss_unweighted",
    "val/recon_MSE",
    "val/recon_RMSE",
    "val/recon_MAE",
    "val/recon_PCC",
    "val/label_MSE",
    "val/label_RMSE",
    "val/label_MAE",
    "val/label_PCC",
    "val/label_F1_weighted",
    "val/label_F1_macro",
    "val/label_F1_micro",
    "val/label_AUROC",
    "val/label_Accuracy",
    "val/label_Precision",
    "val/label_Recall",
]

title_metric_mapping = {
    "jsd": "1 - JSD",
    "mae": "1 - MAE",
    "mse": "1 - MSE",
    "pcc": "PCC",
    "weighted_recall": "Weighted Recall",
    "weighted_precision": "Weighted Precision",
    "weighted_f1": "Weighted F1 Score",
    "macro_f1": "Macro F1 Score",
    "micro_f1": "Micro F1 Score",
    "auprc": "AUPRC",
    "auroc": "AUROC",
    "accuracy": "ACC",
}
title_task_mapping = {
    "multitask": "Multitask",
    "multitask_noguide": "Multitask (no SGG)",
    "focus_recon": "Focus on reconstruction",
    "focus_label": "Focus on labelling",
}
title_colnameC2_mapping = {
    "task": "Task",
    "metric": "Metric",
    "value": "Value",
}
