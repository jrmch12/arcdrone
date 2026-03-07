Build madrona mjx env

mamba create -n madmjx python=3.11
mamba activate madmjx
git submodule update --init --recursive
sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev mesa-common-dev
mamba config append channels nvidia
mamba install cuda -c nvidia/label/cuda-12.4.0
mamba install "cuda-nvcc<=12.5" cudnn cmake
mamba install xorg-xorgproto # yes, for some reason, this has to be installed separately
pip install -U "jax[cuda12_local]<0.6.0" mujoco-mjx mujoco brax
pip install playground #this comes with many tools like mediapy and so
# for the ippynb example:
pip install dm_control
mkdir build && cd build
cmake .. -DCMAKE_LIBRARY_PATH=/lib/x86_64-linux-gnu 
make -j

cd ..

pip install -e .