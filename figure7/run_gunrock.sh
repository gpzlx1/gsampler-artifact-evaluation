# lj 
bin/sage market ~/dataset/lj_loop.mtx --num-runs=4 --device=1 --batch-size=512 --num-children-per-source=25 --num-leafs-per-child=10 --batchnum=947 --remove-self-loops=0 --64bit-SizeT=1 --64bit-ValueT=1 --64bit-VertexT=1
# pr
bin/sage market ~/dataset/pr_loop.mtx --num-runs=4 --device=1 --batch-size=512 --num-children-per-source=25 --num-leafs-per-child=10 --batchnum=385 --remove-self-loops=0 --64bit-SizeT=1 --64bit-ValueT=1 --64bit-VertexT=1