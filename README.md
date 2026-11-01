# BiWorldDB

Code and experiments for **BiWorldDB: Shared-State Analytics over Branching Versions**.

Requires Linux, Python 3.10+, GCC 11+, CMake 3.18+, OpenMP and OpenSSL development headers.
The paper's timing runs used an AMD EPYC 9554 with 503 GiB RAM.

```sh
sudo apt-get install g++ cmake libssl-dev python3-venv
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python reproduce.py check
```

`check` builds the workers, generates a small dataset, compares complete answers with
independent references, and runs randomized branching tests.

```sh
python tools/download_data.py taxi
python reproduce.py benchmark additive --fixture fixtures/taxi2024.bin --repeats 6
python reproduce.py benchmark compiler --fixture fixtures/taxi2024.bin --states 300000 --threads 32
python reproduce.py benchmark mixed
python reproduce.py benchmark holistic
python reproduce.py benchmark rectangles
python reproduce.py benchmark csr
python reproduce.py benchmark planning
python reproduce.py benchmark branch --fixture fixtures/taxi2024.bin
python reproduce.py benchmark history --fixture fixtures/retail2009.bin
python reproduce.py external --fixture fixtures/taxi2024.bin
```

`download_data.py` also accepts `intel`, `air` and `retail`. External comparisons use
Rust and the committed Cargo locks; `external --engines` selects DD, DBSP, DuckDB,
ClickHouse (`chdb`), Polars or DataFusion with their Python packages installed.
`benchmark --help` lists workload options;
`--small` runs a compact configuration. Rectangle experiments at 8,192 coordinates
use about 33 GiB RAM for the dense candidate.

`src/` contains executors, indexes, workload generators and references; `tools/`
contains data converters and the frozen planner; `policies/` contains policies and
experiment cells. Recorded measurements are in `results/`; new runs go to `out/`.
Each run records timings, memory and complete-answer hashes. CSR budgets use MiB.

GPU workers build with `cmake -S . -B build/cuda -DBW_CUDA=ON -DBW_STATES=300000`
and `cmake --build build/cuda -j2`. They require CUDA; architecture 89 targets the RTX 4090.
