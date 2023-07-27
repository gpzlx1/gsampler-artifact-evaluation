
cd deepwalk
echo "===deepwalk"
bash run_dgl.sh
bash run_matrix.sh
bash run_pyg.sh

cd ../node2vec
echo "===node2vec"
bash run_dgl.sh
bash run_gsampler.sh


cd ../graphsage
echo "===graphsage"
bash run_dgl.sh
bash run_gsampler.sh
bash run_pyg.sh

cd ..
echo "===gunrock"
bash run_gunrock.sh
echo "===skywalker"
bash run_skywalker.sh 1>>outputs/skywalker.log 2>>outputs/skywalker.err



