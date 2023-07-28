echo "=====DGL========="
python node2vec_dgl.py --batchsize=1024 --dataset=livejournal 
python node2vec_dgl.py --batchsize=1024 --dataset=products
python node2vec_dgl.py --batchsize=1024 --dataset=papers100m 
python node2vec_dgl.py --batchsize=1024 --dataset=friendster 