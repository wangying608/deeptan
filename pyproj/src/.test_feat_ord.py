# from frn.utils.data_ncv import MyDataModule4Uni
import frn.constants as MC
from frn.utils.uni import CollectFitLog
from frn.dem.explain import DEMFeatureRanking


if __name__ == "__main__":
    # litdata_dir = "/mnt/hdd1/wuch/cotton/run_dem/all_traits/litdata_each2000/ncv_test_0_val_2/val"
    ncv_litdata_dir = "/mnt/hdd1/wuch/cotton/run_dem/all_traits/litdata"
    fit_log_dir = "/mnt/hdd1/wuch/cotton/run_dem/all_traits/models"

    # _feat_rank = DEMFeatureRanking()
    # _feat_rank.run_a_outer(ncv_litdata_dir, fit_log_dir, 0, )

    _clog = CollectFitLog(fit_log_dir)
    _scalar_data = _clog.read_tensorboard_events("/mnt/hdd1/wuch/cotton/run_dem/all_traits/models/train_20241007034929_qJAmTNN/run_ncv_2_2/PVkhdnO/version_0")
    print(_scalar_data, "\n")
