mkdir -p outputs/data

python train_dgl.py > outputs/dgl.txt
python train_pyg.py > outputs/pyg.txt
python train_gsampler.py > outputs/gsampler.txt