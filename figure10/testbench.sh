mkdir -p outputs

conda activate gsampler-ae

cd graphsage/
bash testbench.sh

cd ../ladies/
bash testbench.sh

cd ..

python plot.py
