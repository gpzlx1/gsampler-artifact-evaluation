echo "====graphsage===="
echo "====livejournal===="
skywalker/build/skywalker --bias=0 --ol=1 --input /home/ubuntu/dataset/lj_loop.gr --gmgraph=true --hmgraph=false --umgraph=false --sage --n=512 --ngpu=1 --peritr=1 --batchnum=947 1>>outputs/skywalker.log 2>>outputs/skywalker.err
echo "====products===="
skywalker/build/skywalker --bias=0 --ol=1 --input /home/ubuntu/dataset/pr_loop.gr --gmgraph=true --hmgraph=false --umgraph=false --sage --n=512 --ngpu=1 --peritr=1 --batchnum=385 1>>outputs/skywalker.log 2>>outputs/skywalker.err
echo "====papers100m===="
skywalker/build/skywalker --bias=0 --ol=1 --input /home/ubuntu/dataset/papers100m_with_loop.gr --gmgraph=false --hmgraph=false --umgraph=true --sage --n=512 --ngpu=1 --peritr=1 --batchnum=2358 1>>outputs/skywalker.log 2>>outputs/skywalker.err
echo "====friendster===="
skywalker/build/skywalker --bias=0 --ol=1 --input /home/ubuntu/dataset/friendster_with_loop.gr --gmgraph=false --hmgraph=false --umgraph=true --sage --n=512 --ngpu=1 --peritr=1 --batchnum=12816 1>>outputs/skywalker.log 2>>outputs/skywalker.err



echo "====deepwalk===="
echo "====livejournal===="
skywalker/build/skywalker --bias=0 --ol=1 --input /home/ubuntu/dataset/lj_loop.gr --gmgraph=true --hmgraph=false --umgraph=false --rw=1 --k 1 --d 80 --n=1024 --ngpu=1  --batchnum=474 1>>outputs/skywalker.log 2>>outputs/skywalker.err
echo "====products===="
skywalker/build/skywalker --bias=0 --ol=1 --input /home/ubuntu/dataset/pr_loop.gr --gmgraph=true --hmgraph=false --umgraph=false --rw=1 --k 1 --d 80 --n=1024 --ngpu=1  --batchnum=193 1>>outputs/skywalker.log 2>>outputs/skywalker.err
echo "====papers100m===="
skywalker/build/skywalker --bias=0 --ol=1 --input /home/ubuntu/dataset/papers100m_with_loop.gr --gmgraph=false --hmgraph=false --umgraph=true --rw=1 --k 1 --d 80 --n=1024 --ngpu=1  --batchnum=1179 1>>outputs/skywalker.log 2>>outputs/skywalker.err
echo "====friendster===="
skywalker/build/skywalker --bias=0 --ol=1 --input /home/ubuntu/dataset/friendster_with_loop.gr --gmgraph=false --hmgraph=false --umgraph=true  --rw=1 --k 1 --d 80 --n=1024 --ngpu=1 --batchnum=6408 1>>outputs/skywalker.log 2>>outputs/skywalker.err


echo "====node2vec===="

echo "====livejournal===="
skywalker/build/skywalker --bias=0 --ol=1 --input /home/ubuntu/dataset/lj_loop.gr --gmgraph=true --hmgraph=false  --node2vec --rw=1 --k 1 --d 80 --n=1024 --ngpu=1  --batchnum=474 1>>outputs/skywalker.log 2>>outputs/skywalker.err
echo "====products===="
skywalker/build/skywalker --bias=0 --ol=1 --input /home/ubuntu/dataset/pr_loop.gr --gmgraph=true --hmgraph=false  --node2vec --rw=1 --k 1 --d 80 --n=1024 --ngpu=1 --batchnum=193 1>>outputs/skywalker.log 2>>outputs/skywalker.err

echo "====papers100m===="
skywalker/build/skywalker --bias=0 --ol=1 --input /home/ubuntu/dataset/papers100m_with_loop.gr --gmgraph=false --hmgraph=false --umgraph=true --node2vec --rw=1 --k 1 --d 80 --n=1024 --ngpu=1 --batchnum=1179 1>>outputs/skywalker.log 2>>outputs/skywalker.err
echo "====friendster===="
skywalker/build/skywalker --bias=0 --ol=1 --input /home/ubuntu/dataset/friendster_with_loop.gr --gmgraph=false --hmgraph=false --umgraph=true -node2vec  --rw=1 --k 1 --d 80 --n=1024 --ngpu=1  --batchnum=6408 1>>outputs/skywalker.log 2>>outputs/skywalker.err

grep -Eo 'avg epoch time:[0-9]+\.[0-9]+' outputs/skywalker.log | awk -F':' 'BEGIN{
    OFS=","; 
    }
    {sub(/ avg epoch time:/,"",$0); 
    x=int(NR%4); 
    dataset="unknown"
    if(x==1)
        dataset="livejournal";
    if(x==2)
        dataset="products";
    if(x==3)
        dataset="papers100m";
    if(x==0)
        dataset="friendster";
    algorithm="sage"; 
    if(NR>4) 
    algorithm="rw"; 
    if(NR>8) 
    algorithm="node2vec"; 
    printf "SkyWalker,%s,%s,%s\n", 
    dataset, $2, algorithm}' >> outputs/result.csv