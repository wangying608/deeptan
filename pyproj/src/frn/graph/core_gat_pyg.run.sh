#!bin/bash

cd /home/wuch/Downloads/.tmp/runs
rm -rf /home/wuch/Downloads/.tmp/runs/*
rm -rf /home/wuch/Downloads/.tmp/logs/*

/home/wuch/miniforge3/envs/xnpyg/bin/python /home/wuch/prjs/XRN2P/graph2pheno/pyproj/src/frn/graph/core_gat_pyg.py > /home/wuch/Downloads/.tmp/logs/core_gat_pyg.log
