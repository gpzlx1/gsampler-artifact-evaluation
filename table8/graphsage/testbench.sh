mkdir -p outputs/data

python train_dgl.py --num_epoch=$training_epochs > outputs/dgl.txt
python train_pyg.py --num_epoch=$training_epochs > outputs/pyg.txt
python train_gsampler.py --num_epoch=$training_epochs > outputs/gsampler.txt