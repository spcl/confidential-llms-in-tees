FROM intelanalytics/ipex-llm-cpu:2.2.0-SNAPSHOT

RUN apt install -y gcc g++ cmake

# RUN pip install fastrag
# RUN pip install fastrag[intel]
RUN pip install beir
# RUN pip install git+https://github.com/facebookresearch/KILT.git
# RUN pip install fastrag[elastic]
# RUN sed -i "s/model\.training/getattr(model, 'training', False)/g" /usr/local/lib/python3.11/dist-packages/intel_extension_for_pytorch/transformers/optimize.py

# WORKDIR /app/beir/examples/benchmarking
# RUN git clone https://github.com/IntelLabs/fastRAG.git
# WORKDIR /workspace/fastRAG/benchmarks/BEIR

#CMD ["python3", "msmarco-bm25-sbert.py"]