mkdir -p outputs
:> outputs/result.csv

{
python dgl_gpu.py --dataset=livejournal --num_epoch=$epochs
python dgl_gpu.py --dataset=ogbn-products --num_epoch=$epochs
python dgl_gpu.py --dataset=ogbn-papers100M --use-uva=True --device=cpu --num_epoch=$epochs
python dgl_gpu.py --dataset=friendster --use-uva=True --device=cpu --num_epoch=$epochs
} > outputs/dgl_gpu.txt

{
python gsampler.py --dataset=livejournal --num_epoch=$epochs
python gsampler.py --dataset=ogbn-products --num_epoch=$epochs
python gsampler.py --dataset=ogbn-papers100M --use-uva=True --device=cpu --num_epoch=$epochs
python gsampler.py --dataset=friendster --use-uva=True --device=cpu --num_epoch=$epochs
} > outputs/gsampler.txt

{
python dgl_gpu.py --dataset=livejournal --device=cpu --num_epoch=$epochs
python dgl_gpu.py --dataset=ogbn-products --device=cpu --num_epoch=$epochs
python dgl_gpu.py --dataset=ogbn-papers100M --device=cpu --num_epoch=$epochs
python dgl_gpu.py --dataset=friendster --device=cpu --num_epoch=$epochs
} > outputs/dgl_cpu.txt

{
python pyg_cpu.py --dataset=livejournal --num_epoch=$epochs
python pyg_cpu.py --dataset=ogbn-products --num_epoch=$epochs
python pyg_cpu.py --dataset=ogbn-papers100M --num_epoch=$epochs
} > outputs/pyg_cpu.txt