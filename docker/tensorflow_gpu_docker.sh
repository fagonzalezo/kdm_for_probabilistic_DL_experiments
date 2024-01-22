#!/bin/sh
sudo docker run --rm --runtime=nvidia -it --gpus all -p 8811:8811 -v /NeurIPS-2023/:/NeurIPS-2023/ -v /opt/data:/opt/data rlx/tf
