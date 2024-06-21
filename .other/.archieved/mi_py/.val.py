import h5py


path_h5 = "/home/wuch/prjs/XRN2P/frn/graph_init_20240531175640.h5"

# Open the HDF5 file in read mode
with h5py.File(path_h5, "r") as f:
    # List all datasets in the file
    print(list(f))
    print(f['featnames'])
    print(f['sortedfeatpairs'])
    print(f['sortedmi'])
    print(f['lowstdfeatnames'])
