FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy pandas scipy scikit-image scikit-learn tables pyyaml \
    opencv-python-headless pyTelegramBotAPI skan

# DeepTangleCrawl (JAX CPU) for pipeline/dtc_infer.py — models are mounted at /models/dtc (see docker-compose.yml)
RUN pip install --no-cache-dir "jax[cpu]==0.4.30" "dm-haiku==0.0.12" "numba>=0.60" chex

WORKDIR /app
COPY bot.py /app/
COPY pipeline/ /app/pipeline/

ENV WORK_DIR=/work
CMD ["python", "bot.py"]
