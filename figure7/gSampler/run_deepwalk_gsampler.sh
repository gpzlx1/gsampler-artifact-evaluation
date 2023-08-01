echo "=====gsampler======="
python deepwalk_gsampler.py --batchsize=1024 --dataset=livejournal --big-batch=51200
python deepwalk_gsampler.py --batchsize=1024 --dataset=products --big-batch=51200
python deepwalk_gsampler.py --batchsize=1024 --dataset=papers100m --device=cpu --use-uva --big-batch=5120
python deepwalk_gsampler.py --batchsize=1024 --dataset=friendster --device=cpu --use-uva --big-batch=5120