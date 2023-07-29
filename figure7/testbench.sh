source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate gsampler-ae
echo "===test gSampler==="
cd gSampler
bash run_deepwalk_gsampler.sh
bash run_node2vec_gsampler.sh
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
conda activate rapids-23.02
cd cugraph
bash run_cugraph.sh
conda activate gsampler-ae


