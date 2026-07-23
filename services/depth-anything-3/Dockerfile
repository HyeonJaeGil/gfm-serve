ARG COMMON_BASE_IMAGE=gfm-serve-core:latest
FROM ${COMMON_BASE_IMAGE}

ENV GFM_SERVE_BACKEND=depth-anything-3

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1 torchvision==0.19.1

RUN python -m pip install \
        numpy==1.26.4 \
        scipy==1.14.1 \
        xformers==0.0.28 \
        addict \
        einops \
        e3nn \
        evo \
        huggingface-hub \
        imageio \
        moviepy==1.0.3 \
        omegaconf \
        opencv-python-headless==4.11.0.86 \
        pillow-heif \
        plyfile \
        pycolmap \
        safetensors \
        trimesh

COPY services/depth-anything-3 ./services/depth-anything-3

RUN python -m pip install --no-deps ./services/depth-anything-3/upstream \
    && python -m pip install ./services/depth-anything-3

RUN python /app/services/depth-anything-3/scripts/smoke_import.py
