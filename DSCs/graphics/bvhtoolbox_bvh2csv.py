from bvhtoolbox.convert import bvh2csv
from glob import glob

for num in range(11,145):
    out_dir = 'D:\\CMU_bvh\\'+str(num)+'\\BVH\\'
    file_dir = out_dir+'*.bvh'
    dirs= glob(file_dir)

    bvh2csv(dirs,dst_dirpath=out_dir,export_hierarchy=False,export_rotation=False)