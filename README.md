# RetinaFace in PyTorch

conda create -n retinaface_env python=3.9

# 2. Activate the new environment
conda activate retinaface_env

# 3. Install PyTorch and TorchVision (and CUDA support)
# This is the recommended command from the PyTorch website.
# Please check pytorch.org for the command that matches your specific CUDA version.
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. Install the remaining dependencies using pip
pip install numpy opencv-python cython
