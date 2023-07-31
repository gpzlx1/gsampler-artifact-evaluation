mkdir -p outputs

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate gsampler-ae

cd ladies/
bash testbench.sh

cd ../asgcn/
bash testbench.sh

cd ../pass/
bash testbench.sh

cd ../shadowkhop/
bash testbench.sh

cd ..

python plot.py
