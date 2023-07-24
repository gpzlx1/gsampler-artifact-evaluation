cd graphsage

python graphsage_matrix.py --batchsize=512 --dataset=products
python graphsage_matrix.py --batchsize=512 --dataset=livejournal
python graphsage_matrix.py --batchsize=512 --dataset=papers100m --device=cpu --use-uva
python graphsage_matrix.py --batchsize=512 --dataset=friendster --device=cpu --use-uva