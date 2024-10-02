import os
import sys
from frn.s2g.pipeline import SNP2GBTransPipe


if len(sys.argv) < 3:
    print('Start default NCV: 10 outer folds and 5 inner folds')
    list_ncv = [[i,j] for i in range(10) for j in range(5)]
else:
    print('Start with NCV: {} outer folds and {} inner folds'.format(sys.argv[1], sys.argv[2]))
    list_ncv = [[int(sys.argv[1]), int(sys.argv[2])]]

work_dir_home = '/mnt/hdd2/homext/wuch/xn2p'
litdata_dir = os.path.join(work_dir_home, "data", "litdata", "s2g", "ath", "ft16")

log_dir = os.path.join(work_dir_home, "run", "s2g", "log_local")
output_dir = os.path.join(work_dir_home, "run", "s2g", "converted")


if __name__ == '__main__':
    pipe_trans = SNP2GBTransPipe(
        dir_log=log_dir,
        dir_output=output_dir,
        overwrite_collected_log=True,
    )
    pipe_trans.collect_models()

    pipe_trans.convert_snp(
        dir_litdata=litdata_dir,
        list_ncv=list_ncv,
    )
