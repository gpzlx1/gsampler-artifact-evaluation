mkdir -p outputs

cd graphsage/
bash testbench.sh

cd ../ladies/
bash testbench.sh

cd ..

python plot.py