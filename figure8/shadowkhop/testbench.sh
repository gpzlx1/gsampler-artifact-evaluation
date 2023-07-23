mkdir -p outputs

python dgl_gpu.py --dataset=livejournal > outputs/dgl_gpu.txt
python dgl_gpu.py --dataset=ogbn-products >> outputs/dgl_gpu.txt
python dgl_gpu.py --dataset=ogbn-papers100M --use-uva=True --device=cpu >> outputs/dgl_gpu.txt
python dgl_gpu.py --dataset=friendster --use-uva=True --device=cpu >> outputs/dgl_gpu.txt

python gsampler.py --dataset=livejournal > outputs/gsampler.txt
python gsampler.py --dataset=ogbn-products >> outputs/gsampler.txt
python gsampler.py --dataset=ogbn-papers100M --use-uva=True --device=cpu >> outputs/gsampler.txt
python gsampler.py --dataset=friendster --use-uva=True --device=cpu >> outputs/gsampler.txt

python dgl_gpu.py --dataset=livejournal --device=cpu > outputs/dgl_cpu.txt
python dgl_gpu.py --dataset=ogbn-products --device=cpu >> outputs/dgl_cpu.txt
python dgl_gpu.py --dataset=ogbn-papers100M --device=cpu >> outputs/dgl_cpu.txt
python dgl_gpu.py --dataset=friendster --device=cpu >> outputs/dgl_cpu.txt

python pyg_cpu.py --dataset=livejournal > outputs/pyg_cpu.txt
python pyg_cpu.py --dataset=ogbn-products >> outputs/pyg_cpu.txt
python pyg_cpu.py --dataset=ogbn-papers100M >> outputs/pyg_cpu.txt