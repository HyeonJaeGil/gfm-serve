ARG COMMON_BASE_IMAGE=gfm-serve-core:latest
FROM ${COMMON_BASE_IMAGE}

COPY services/depth-anything-3 ./services/depth-anything-3

ENV GFM_SERVE_BACKEND=depth-anything-3

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install ./services/depth-anything-3/upstream \
    && python -m pip install ./services/depth-anything-3

RUN python /app/services/depth-anything-3/scripts/smoke_import.py
