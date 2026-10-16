FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

WORKDIR /benchmark

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ool_core/ ool_core/
COPY benchmark.py reference_checksums.json ./
COPY data/ data/

VOLUME /output

ENTRYPOINT ["python", "benchmark.py"]
