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

splits = [abbr_train, abbr_val, abbr_test]

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
    "spearman": "Spearman",
    "weighted_recall": "Recall (weighted)",
    "weighted_precision": "Precision (weighted)",
    "weighted_f1": "F1 Score (weighted)",
    "macro_f1": "F1 Score (macro)",
    "micro_f1": "F1 Score (micro)",
    "auprc": "AUPRC",
    "auroc": "AUROC",
    "accuracy": "ACC",
    "kbet_true_label": "kBET (true labels)",
    "kbet_pred_label": "kBET (predicted labels)",
    "asw_true_label": "ASW (true labels)",
    "asw_pred_label": "ASW (predicted labels)",
    "kbet": "kBET",
    "asw": "ASW",
    "ari": "ARI",
    "nmi": "NMI",
    "ami": "AMI",
    "ari_leiden": "ARI (Leiden)",
    "nmi_leiden": "NMI (Leiden)",
    "ami_leiden": "AMI (Leiden)",
    "method": "Method",
    "train_size": "Train size",
    "n_feat": "Number of features for test",
}
title_task_mapping = {
    "multitask": "Multitask",
    "multitask_noguide": "Multitask (no SGG)",
    "focus_recon": "Focus on reconstruction",
    "focus_label": "Focus on labelling",
    "focus_label_on_focus_recon": "Fine-tuned on reconstruction",
}
title_colnameC2_mapping = {
    "task": "Task",
    "metric": "Metric",
    "value": "Value",
}
