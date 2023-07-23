mkdir -p outputs

python test.py > outputs/ogbn_products.txt
python test.py --dataset=ogbn-papers100M --device=cpu --use-uva=True > outputs/ogbn_papers100M.txt