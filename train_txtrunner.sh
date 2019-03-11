export PATH=~/anaconda3/bin:$PATH
export PATH=/usr/local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cudnn7.0-9.0/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cudnn7.0-9.0/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cudnn7.0-9.0/lib64:$LIBRARY_PATH
python train_softmax_Only_TXT_softmaxF1.py
