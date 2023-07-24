echo "====DGL========="
echo "====livejounal========="
python deepwalk_dgl.py --batchsize=1024 --dataset=livejournal 
echo "====products========="
python deepwalk_dgl.py --batchsize=1024 --dataset=products
echo "====papers100m========"
python deepwalk_dgl.py --batchsize=1024 --dataset=papers100m --device=cpu --use-uva
echo "====friendster========"
python deepwalk_dgl.py --batchsize=1024 --dataset=friendster --device=cpu --use-uva

