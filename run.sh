epochs=2
export epochs

training_epochs=2
export training_epochs

cd figure8
bash testbench.sh

cd ../figure10
bash testbench.sh

cd ../table8
bash testbench.sh

