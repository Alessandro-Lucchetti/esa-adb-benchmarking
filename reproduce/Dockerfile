# Environment of scripts/reproduce.py. The repository is mounted at /work at run time, so
# the image holds only Python and the pinned packages; reproduce/run.sh builds and runs it.
#
#   docker build -t esa-adb-reproduce reproduce
#   docker run --rm -v "$PWD":/work --user "$(id -u):$(id -g)" esa-adb-reproduce [stage ...]
FROM python:3.10.21-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system safe.directory /work

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# writable locations for any --user; the repository code is imported from the mount
ENV PYTHONPATH=/work/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib \
    MPLBACKEND=Agg

WORKDIR /work
ENTRYPOINT ["python", "scripts/reproduce.py"]
CMD ["all"]
