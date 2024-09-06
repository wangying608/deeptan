import os
import sys
from frn.s2g.pipeline import SNP2GBTransPipe


list_ncv = [[int(sys.argv[1]), int(sys.argv[2])]]
work_dir_home = '/mnt/hdd2/homext/wuch/xn2p'
log_dir = os.path.join(work_dir_home, 's2g_models')
output_dir = os.path.join(work_dir_home, 's2g_trans')
litdata_dir = os.path.join(work_dir_home, 'data/_optimized/SNP_zsLabel_FT16_except_external_val')
bsize = 256

if __name__ == '__main__':
    pipe_trans = SNP2GBTransPipe(
        dir_log=log_dir,
        dir_output=output_dir,
        overwrite_collected_log=True,
    )
    pipe_trans.collect_trained_models()

    pipe_trans.convert_snp(
        dir_litdata=litdata_dir,
        list_ncv=list_ncv,
        batch_size=bsize,
    )
