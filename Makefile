# Get version of CUDA and enable it for compilation if CUDA > 11.0
# This solves https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/53
# and https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/84
# when running in Docker
# Check if nvcc is installed
NVCC := $(shell which nvcc)
ifeq ($(NVCC),)
	# NVCC not found
	USE_CUDA := 0
	NVCC_VERSION := "not installed"
else
	NVCC_VERSION := $(shell nvcc --version | grep -oP 'release \K[0-9.]+')
	USE_CUDA := $(shell echo "$(NVCC_VERSION) > 11" | bc -l)
endif

# Add the list of supported ARCHs
ifeq ($(USE_CUDA), 1)
	TORCH_CUDA_ARCH_LIST := "3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
	BUILD_MESSAGE := "I will try to build the image with CUDA support"
else
	TORCH_CUDA_ARCH_LIST :=
	BUILD_MESSAGE := "CUDA $(NVCC_VERSION) is not supported"
endif

build-image:
	@echo $(BUILD_MESSAGE)
	docker build --build-arg USE_CUDA=$(USE_CUDA) \
	--build-arg TORCH_ARCH=$(TORCH_CUDA_ARCH_LIST) \
	-t gaussianformer:latest .
	docker run -d --gpus all -it --rm --net=host \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v "${PWD}":/home/appuser/GaussianFormer \
	-w /home/appuser/GaussianFormer \
	-e DISPLAY=$(DISPLAY) \
	--name=gaussianformer \
	--ipc=host gaussianformer:latest
	docker exec -it gaussianformer sh -c "cd model/encoder/gaussian_encoder/ops && pip install --no-build-isolation -e ."
	docker exec -it gaussianformer sh -c "cd model/head/localagg && pip install --no-build-isolation -e ."
	docker exec -it gaussianformer sh -c "cd model/head/localagg_prob && pip install --no-build-isolation -e ."
	docker exec -it gaussianformer sh -c "cd model/head/localagg_prob_fast && pip install --no-build-isolation -e ."
	docker commit gaussianformer gaussianformer:latest
	docker stop gaussianformer


run:
	docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$(DISPLAY) -e USER=$(USER) \
	-e runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all \
	-e PYTHONPATH=/home/appuser/GaussianFormer \
	-v "${PWD}":/home/appuser/GaussianFormer \
	-w /home/appuser/GaussianFormer \
	-v "${SCANNET_PATH}":/data \
	--shm-size 128G \
	--net host --gpus all --privileged --name gaussianformer gaussianformer:latest /bin/bash

# cd /home/appuser/GaussianFormer/model/encoder/gaussian_encoder/ops && pip install -v --no-build-isolation -e .
# cd /home/appuser/GaussianFormer/model/head/localagg && pip install -v --no-build-isolation -e .
# cd /home/appuser/GaussianFormer/model/head/localagg_prob && pip install -v --no-build-isolation -e .
# cd /home/appuser/GaussianFormer/model/head/localagg_prob_fast && pip install -v --no-build-isolation -e .

# docker run -d --gpus all -it --rm --net=host \
# -v /tmp/.X11-unix:/tmp/.X11-unix \
# -v "${PWD}":/home/appuser/GaussianFormer \
# -w /home/appuser/GaussianFormer \
# -e DISPLAY=$(DISPLAY) \
# --name=gaussianformer \
# --ipc=host gaussianformer:latest
# #docker exec -it gaussianformer sh -c "cd model/encoder/gaussian_encoder/ops && pip install -e ."
# 3docker exec -it gaussianformer sh -c "cd model/head/localagg && pip install -e ."
# docker commit gaussianformer gaussianformer:latest
# docker stop gaussianformer
# docker exec -it gaussianformer sh -c "cd model/head/localagg_prob && pip install -e ."
# docker exec -it gaussianformer sh -c "cd model/head/localagg_prob_fast && pip install -e ."

# export SCANNET_PATH=/media/sequor/PortableSSD/scannetpp && make run
# python train.py --py-config config/scannetpp_gs144000.py --work-dir out/scannetpp
# python train.py --py-config config/scannetpp_gs25600_solid.py --work-dir out/scannetpp
# python train.py --py-config config/scannetppsmall_gs25600_solid.py --work-dir out/scannetpp
# CUDA_VISIBLE_DEVICES=0 python visualize_scannetpp.py --py-config config/scannetppsmall_gs25600_solid.py --work-dir out/scannetppsmall --vis-occ --model-type base --resume-from out/scannetppsmall/epoch_200.pth
# CUDA_VISIBLE_DEVICES=0 python visualize_scannetpp.py --py-config config/scannetpp_gs25600_solid.py --work-dir out/scannetpp --vis-occ --model-type base --resume-from out/scannetpp/epoch_30.pth

# export SCANNET_PATH=/data/scannetpp && make run
# python train.py --py-config config/scannetpp_gs144000.py --work-dir out/scannetpp
# python train.py --py-config config/scannetppsmall_gs144000_solid.py --work-dir out/scannetpp


# userful commands
# xhost +Local:*  && xhost
# sudo chown -R $USER: $HOME