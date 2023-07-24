echo "graphsage_matrix"
python graphsage_matrix.py --dataset livejournal --batchsize=512 --big-batch=51200
python graphsage_matrix.py --dataset products --batchsize=512 --big-batch=51200
python graphsage_matrix.py --dataset=papers100m --batchsize=512 --data-type=long --use-uva --device=cpu --big-batch=10240
python graphsage_matrix.py --dataset=friendster --batchsize=512 --data-type=long --use-uva --device=cpu --big-batch=5120

