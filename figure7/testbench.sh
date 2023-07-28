echo "===test gSampler==="
cd gSampler
bash run_deepwalk_gsampler.sh
bash node2vec_gsampler.py
bash run_graphsage_gsampler.sh

echo "===test DGL==="
cd ../dgl
bash run_deepwalk_dgl.sh
bash run_node2vec_dgl.sh
bash run_graphsage_dgl.sh

echo "===test PyG==="
cd ../PyG
bash run_deepwalk_pyg.py
bash run_graphsage_pyg.sh

cd ..

echo "===test gunrock==="
bash run_gunrock.sh

echo "===test skywalker==="
bash run_skywalker.sh 

echo "===test cuGraph==="
cd curgraph
bash run_cugraph.sh


