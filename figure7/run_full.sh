# echo "executing skywalker"
# cd /home/ubuntu/newsky/
# bash run.sh
cd /home/ubuntu/gs-experiments/main_exp_sampling/simple_algorithms/
echo "executing matrix and dgl graphsage"
cd graphsage
bash run.sh
echo "executing matrix and dgl deepwalk"
cd ../deepwalk
bash run.sh
# echo "executing matrix and dgl node2vec"
# cd ../node2vec
# bash run.sh
