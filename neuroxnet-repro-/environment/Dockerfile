FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
WORKDIR /work
COPY environment/requirements.txt /work/environment/requirements.txt
RUN pip install --no-cache-dir -r environment/requirements.txt
COPY . /work
ENV PYTHONPATH=/work/src
RUN pytest -q tests/
ENTRYPOINT ["bash"]
