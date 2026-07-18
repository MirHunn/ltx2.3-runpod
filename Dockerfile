# torch 2.8 + CUDA 12.8 (Blackwell-ready) — satisfies ltx-core's torch~=2.7 pin.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Official LTX-2 inference code. Install ltx-core FIRST (local, newest) so the
# ltx-pipelines dependency resolves to it instead of the stale PyPI release.
# ltx-kernels is deliberately NOT installed: only needed for multi-GPU / FP8.
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /opt/LTX-2 \
    && pip install --no-cache-dir /opt/LTX-2/packages/ltx-core \
    && pip install --no-cache-dir /opt/LTX-2/packages/ltx-pipelines

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
