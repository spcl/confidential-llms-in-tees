numactl -C 60-119 -m 1 python3 beir/examples/benchmarking/benchmark_bm25.py
numactl -C 60-119 -m 1 python3 beir/examples/benchmarking/benchmark_sbert.py
numactl -C 60-119 -m 1 python3 beir/examples/benchmarking/benchmark_bm25_ce_reranking.py