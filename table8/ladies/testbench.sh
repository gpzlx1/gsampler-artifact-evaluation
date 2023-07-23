mkdir -p outputs/data

python train_dgl.py > outputs/dgl.txt
python train_gsampler.py > outputs/gsampler.txt