ARG COMMON_BASE_IMAGE=gfm-serve-core:latest
FROM ${COMMON_BASE_IMAGE}

COPY services/vggt ./services/vggt

ENV GFM_SERVE_BACKEND=vggt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.3.1 torchvision==0.18.1 \
    && python -m pip install ./services/vggt/upstream ./services/vggt
