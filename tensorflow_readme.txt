To install tensorflow with gpu support

1. Install nvidia drivers
2. Install nvidia cuda from rpm
nvidia cuda muste be 9.0 and cuDNN 7.0

cuda-repo-opensuse422-9-0-local-9.0.176-1.x86_64

add to the .bashrc or .profile file
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64\ {LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

or instead "it is recommended to make language settings in ~/.profile"
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=/usr/local/cuda/include:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=/usr/local/cuda:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH

2. Install cuDNN
nvidia cuda muste be 9.0 and cuDNN 7.0
cudnn-9.0-linux-x64-v7 is for cuda 9.0

tar xvzf cudnn-9.0-linux-x64-v7.tgz
sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

3. Install libcupti-dev for nvidia<=7.x
7. Run
   pip2 install tensorflow==1.8
8. With bumblebee use optirun
   -set the user in the bumblebee,video group
