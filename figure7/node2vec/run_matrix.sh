echo "=======Matrix========="
python node2vec_matrix.py --batchsize=1024 --dataset=livejournal --big-batch=20480
python node2vec_matrix.py --batchsize=1024 --dataset=products  --big-batch=20480
python node2vec_matrix.py --batchsize=1024 --dataset=papers100m --device=cpu --use-uva  --big-batch=4096
python node2vec_matrix.py --batchsize=1024 --dataset=friendster --device=cpu --use-uva  --big-batch=2048


