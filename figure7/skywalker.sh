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