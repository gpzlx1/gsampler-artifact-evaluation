echo "graphsage_dgl"
python graphsage_dgl.py --dataset=livejournal --batchsize=512 
python graphsage_dgl.py --dataset=products --batchsize=512 
python graphsage_dgl.py --dataset=papers100m --batchsize=512  --data-type=long --use-uva --device=cpu
python graphsage_dgl.py --dataset=friendster --batchsize=512  --data-type=long --use-uva --device=cpu