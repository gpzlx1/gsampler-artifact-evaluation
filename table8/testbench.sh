mkdir -p outputs
:> outputs/result.csv

cd graphsage/
bash testbench.sh

cd ../ladies/
bash testbench.sh

cd ..