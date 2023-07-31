epochs=2
export epochs

training_epochs=2
export training_epochs



echo "run figure7"
cd figure7/
bash testbench.sh

echo "run figure8"
cd ../figure8
bash testbench.sh

cd ../figure10
bash testbench.sh

