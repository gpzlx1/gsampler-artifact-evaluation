mkdir -p outputs
:> outputs/result.csv

python test.py --num_epoch=$epochs > outputs/ogbn_products.txt
python test.py --dataset=ogbn-papers100M --device=cpu --use-uva=True --num_epoch=$epochs > outputs/ogbn_papers100M.txt