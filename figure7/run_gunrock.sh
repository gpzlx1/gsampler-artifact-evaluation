# lj 
gunrock/build/bin/sage market /home/ubuntu/dataset/lj_loop.mtx --num-runs=6 --device=1 --batch-size=512 --num-children-per-source=25 --num-leafs-per-child=10 --batchnum=947 --remove-self-loops=0 --64bit-SizeT=1 --64bit-ValueT=1 --64bit-VertexT=1 1>>outputs/gunrock.log 2>>outputs/gunrock.err
# pr
gunrock/build/bin/sage market /home/ubuntu/dataset/pr_loop.mtx --num-runs=6 --device=1 --batch-size=512 --num-children-per-source=25 --num-leafs-per-child=10 --batchnum=385 --remove-self-loops=0 --64bit-SizeT=1 --64bit-ValueT=1 --64bit-VertexT=1 1>>outputs/gunrock.log 2>>outputs/gunrock.err

grep -Eo 'avg\. elapsed: [0-9.]+ ms' output/gunrock.log | awk -F': ' 'BEGIN{OFS=","; print "baseline,dataset,execution time,algorithm time"}{sub(/ ms/,"",$2); if (NR==1) {printf "gunrock,livejournal,%s,sage\n", $2} else {printf "gunrock,products,%s,sage\n", $2}}' > output/gunrock.csv

