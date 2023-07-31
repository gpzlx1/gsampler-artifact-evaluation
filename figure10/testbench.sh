mkdir -p outputs

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate gsampler-ae

cd graphsage/
bash testbench.sh

cd ../ladies/
bash testbench.sh

cd ..

python plot.py
