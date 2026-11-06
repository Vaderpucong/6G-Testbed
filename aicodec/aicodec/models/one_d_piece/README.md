# One-D-Piece (ODP) <-> Qwen3-VL MMBench, lossy UDP link

End-to-end lossy-link image experiment: One-D-Piece L-256 tokenizes MMBench
images, sends them over UDP (chunked or whole-image mode, no retransmission),
the cloud server reconstructs the survivors, and Qwen3-VL scores the
reconstructed folder offline. Only this UDP path is kept; the earlier
HTTP and realtime-WebRTC ODP links were dropped as superseded.

> This file is the ODP doc entry point and lives next to the model code at
> `aicodec/aicodec/models/one_d_piece/README.md`. There is no repo-root
> `One-D-Piece/` folder anymore -- dataset/results live under
> `aicodec/aicodec/dataset/MMBench/`, eval/CLI tooling lives under
> `aitestbed/tools/`, and the model package itself lives here.

## Repo layout (ODP-related parts only)

```
<repo root>/
  run_odp_mmbench_pipeline.py          # one-click pipeline: server -> orchestrator
                                        # (UDP link) -> Qwen3-VL eval -> summary
  requirements-qwen.txt                # isolated pip env for the Qwen3-VL eval
                                        # step (conflicts with aitestbed/requirements.txt:
                                        # torch/transformers pins differ)
  aicodec/aicodec/
    models/
      one_d_piece/                     # self-contained ODP model package
        __init__.py                    # registers OneDPieceBridge as "one_d_piece"
        bridge.py                      # aicodec bridge (enc/dec)
        odp_l256.yaml                  # model config (vq/vit hyperparams)
        odp_udp_wire.py                # shared UDP datagram framing +
                                        # 12-bit token pack/unpack/reassembly
                                        # (import as `from aicodec.models
                                        # .one_d_piece import odp_udp_wire`);
                                        # pure stdlib, no aiortc dependency,
                                        # so the UDP client/server/sweep tool
                                        # use this instead of odp_codec.py
        README.md                      # this file
        vendor/modeling/                vendored One-D-Piece model source
          one_d_piece.py, titok.py, modules/, quantizer/
        checkpoints/
          pytorch_model.bin            # ODP L-256 weights (NOT in git)
      qwen3vl/
        Qwen3-VL-30B-A3B-Instruct/      # Qwen3-VL weights (NOT in git)
    dataset/
      MMBench/                          # eval dataset + results (resource folder,
                                         # not a Python package)
        MMBench_DEV_EN/                 # dataset: manifest.jsonl (in git) +
                                         # images (NOT in git, see below)
        results/                        # score tables written by run_odp_mmbench.sh
  aitestbed/
    tools/
      mmbench_eval.py                 # MMBench MCQ eval driver (Qwen3-VL)
      fetch_odp_assets.sh             # data check + optional weight download
      run_odp_mmbench.sh              # CLI wrapper: ODP-reconstructed/baseline
                                       # folder -> Qwen3-VL accuracy table
      check_odp_image_recon.py        # optional PSNR/SSIM acceptance check
      run_odp_payload_drop_sweep.py   # payload-level token-drop sweep driver
                                       # (see "Payload-level token-drop sweep"
                                       # below)
    server/odp_image_udp_server.py     # cloud: UDP receive + ODP decode + recon
    clients/odp_image_udp_client_{whole,chunked}.py   # edge: ODP encode + UDP send
    scenarios/odp_image_udp.py         # orchestrator scenario (provider "odp-image-udp")
    configs/scenarios.yaml             # odp_image_udp_chunked/whole scenarios
                                        # merged in (see runner_profile_sets:
                                        # odp_image_udp_profiles)
    orchestrator.py                    # registers scenario "odp_image_udp" +
                                        # client provider "odp-image-udp"
```

## Self-contained model package (bridge.py + vendor/ + checkpoints/)

`bridge.py` is self-contained: `setup()` puts `one_d_piece/vendor` on
`sys.path` by default and imports `modeling.one_d_piece` from there, loads
`odp_l256.yaml` next to it, and the checkpoint from `checkpoints/` in the
same package. No external checkout or env vars are required for a normal
run.

| Env var          | Default (bundled)                                                  | Override use case                 |
| ---------------- | ------------------------------------------------------------------ | --------------------------------- |
| `ODP_REPO`       | `aicodec/aicodec/models/one_d_piece/vendor`                        | point at an external ODP checkout |
| `ODP_CONFIG`     | `aicodec/aicodec/models/one_d_piece/odp_l256.yaml`                 | use a different model config      |
| `ODP_CHECKPOINT` | `aicodec/aicodec/models/one_d_piece/checkpoints/pytorch_model.bin` | point at different weights        |
| `ODP_NUM_TOKENS` | `256`                                                              | tokens kept per frame, 1..256     |
| `ODP_DEVICE`     | `cuda` if available, else `cpu`                                    | force CPU                         |

These only need to be set to override the bundled defaults (e.g. testing
against an external One-D-Piece checkout); a normal run needs nothing
exported.

## Data and weights (NOT tracked in git -- migrate manually)

* `aicodec/aicodec/dataset/MMBench/MMBench_DEV_EN/manifest.jsonl` is in git;
  the ~4300 jpgs are NOT (too large / no LFS). Copy them from an existing
  checkout or the team share into
  `aicodec/aicodec/dataset/MMBench/MMBench_DEV_EN/`.
* `aicodec/aicodec/models/one_d_piece/checkpoints/pytorch_model.bin` (ODP
  L-256 tokenizer, ~2.4 GB) and
  `aicodec/aicodec/models/qwen3vl/Qwen3-VL-30B-A3B-Instruct/` (~60 GB) are
  NOT in git. Fetch with:
  ```bash
  bash aitestbed/tools/fetch_odp_assets.sh --weights
  ```
  or place them manually at the paths above.

## Setup

```bash
python -m venv ~/envs/qwen3vl && source ~/envs/qwen3vl/bin/activate
pip install -r requirements-qwen.txt
# installs torch/omegaconf/torchvision for the ODP bridge itself, if not
# already present in your orchestrator env:
pip install torch torchvision omegaconf einops huggingface_hub

bash aitestbed/tools/fetch_odp_assets.sh --weights
```

## Running

```bash
# one-click: server + orchestrator (UDP whole-image scenario) + Qwen3-VL eval
python run_odp_mmbench_pipeline.py \
    --profiles lossy --interface lo --max-images 0 \
    --eval-conda-env qwen3vl --eval-gpu 1

# one shot network-profile sweep (one full 4-phase run per profile)
python run_odp_mmbench_pipeline.py \
    --mode whole --profiles no_emulation lossy 6g_itu_hrllc cell_edge \
    --interface lo

# smoke test with a handful of images before a full run
python run_odp_mmbench_pipeline.py \
    --mode whole --profiles lossy --max-images 20 --eval-limit 20

# or score an existing folder directly
bash aitestbed/tools/run_odp_mmbench.sh --baseline-only --limit 20
bash aitestbed/tools/run_odp_mmbench.sh --summary
```

### `run_odp_mmbench_pipeline.py` CLI reference

**Scenario / data**

| Flag | Default | Notes |
| --- | --- | --- |
| `--config` | `aitestbed/configs/scenarios.yaml` | orchestrator scenario file |
| `--scenario` | `odp_image_udp_whole` | scenario name directly |
| `--mode` | (none) | shortcut for `--scenario odp_image_udp_{mode}` (`whole`/`chunked`); wins over `--scenario` |
| `--max-images` | from yaml `max_images` | cap images for this run (smoke tests) |
| `--interface` | `lo` | NIC netem is applied to (`lo` for loopback, real NIC name otherwise) |
| `--runs` | `1` | repeats per profile |
| `--run-timeout` | `3600.0` s | per-run timeout (full dataset needs > 180s) |

**Network / token-drop**

| Flag | Default | Notes |
| --- | --- | --- |
| `--profiles` | `["lossy"]` | one or more network profiles (see `profiles.yaml`); each profile gets its own isolated 4-phase run |
| `--payload-drop-mode` | (none) | `none`/`tail`/`random`; **whole-mode only**, and only takes effect when the active profile is `lossy_payload` |
| `--payload-drop-pct` | (none) | 0..100, percent of tokens dropped; same `lossy_payload`-only gating |

**Server / GPU**

| Flag | Default | Notes |
| --- | --- | --- |
| `--server-port` | `1237` | ODP UDP server listen port |
| `--odp-device` | empty | explicit decode device (e.g. `cuda:0`); usually just set `CUDA_VISIBLE_DEVICES` |
| `--server-ready-timeout` | `600.0` s | weight warm-up can take minutes on a cold start |
| `--post-send-settle` | `10.0` s | extra wait after send completes so tail packets land |
| `--out-dir` | `results/odp_mmbench_pipeline` | output root (relative paths resolve under `aitestbed/`) |

**Qwen3-VL eval**

| Flag | Default | Notes |
| --- | --- | --- |
| `--skip-eval` | off | only run the link/reconstruction phases, skip Qwen3-VL scoring |
| `--eval-conda-env` | `qwen3vl` | conda env `mmbench_eval.py` runs in (via `conda run`) |
| `--eval-model` | bundled `Qwen3-VL-30B-A3B-Instruct` | custom weights path / hub id |
| `--eval-gpu` | inherits `CUDA_VISIBLE_DEVICES` | GPU visible to the eval subprocess only |
| `--eval-limit` | unlimited | score only the first N samples (smoke test) |
| `--tensor-parallel-size` | `1` | vLLM tensor-parallel degree |
| `--gpu-memory-utilization` | `0.9` | vLLM GPU memory fraction |

### Network profiles (`aitestbed/configs/profiles.yaml`)

Standard 3GPP SA4 (S4-260848) network-impairment profiles live here:
`delay_ms` / `jitter_ms` / `delay_distribution` / `loss_pct` /
`loss_correlation_pct` / `reorder_pct` / `duplicate_pct` / `limit_packets` /
`rate_mbit`. The two directly relevant to this experiment:

| Profile | `loss_pct` | Purpose |
| --- | --- | --- |
| `lossy` | `10.0` | real netem loss (10%), applied via `tc/netem` |
| `lossy_payload` | `0` (zero netem) | dedicated "carrier" profile for the payload-level synthetic drop below; has no effect unless combined with `--payload-drop-mode`/`--payload-drop-pct` |

The rest (`6g_itu_hrllc`, `5g_urban`, `wifi_good`, `cell_edge`,
`satellite_leo`/`geo`, `congested`, `5qi_7`, `5qi_80`, `no_emulation`) are
standard 3GPP operating points usable directly as `--profiles` for a real
network-loss sweep.

### Two ways to lose tokens

| | Real network loss (netem) | Payload-level synthetic drop |
| --- | --- | --- |
| Trigger | `--profiles lossy` (or any non-zero `loss_pct` profile) | `--profiles lossy_payload` + `--payload-drop-mode` + `--payload-drop-pct` |
| Layer | UDP datagram layer; `tc/netem` drops whole datagrams on the NIC | application layer; tokens are dropped **before** 12-bit packing |
| Supported modes | whole and chunked | **whole only** |
| Controllability | random drop location, loss rate not exactly reproducible | exact, reproducible drop ratio and strategy (tail/random) |
| Composable? | No — `lossy_payload` carries zero netem loss on purpose, so the two mechanisms stay orthogonal and never silently stack |

> UDP is fire-and-forget, so the client can only confirm a datagram was
> *sent*. Real delivery / per-chunk loss / whole-image loss counts can only
> come from the server side (`odp_image_udp_server.py`'s `report.json` — see
> "Output layout" below).

## Payload-level token-drop sweep

In addition to real network loss (netem profiles), the UDP clients
(`odp_image_udp_client_whole.py` / `odp_image_udp_client_chunked.py`) can
drop tokens **at the source**, before 12-bit packing, independent of any
link impairment. This is a pure client-side degradation knob
(`aicodec.models.one_d_piece.odp_udp_wire.apply_payload_token_drop`), folded in from the
older offline `run_summary_tail.sh` / `run_summary_random.sh` study so the
same two strategies can be exercised live over the UDP link:

```bash
# tail: truncate the least-important TRAILING tokens (ODP orders tokens by
# importance, so this is the "keep only the first N%" ablation)
python aitestbed/clients/odp_image_udp_client_whole.py \
    --image-dir aicodec/aicodec/dataset/MMBench/MMBench_DEV_EN \
    --server 127.0.0.1:1237 --payload-drop-mode tail --payload-drop-pct 30

# random: zero-fill a random subset of token positions
python aitestbed/clients/odp_image_udp_client_whole.py \
    ... --payload-drop-mode random --payload-drop-pct 30
```

`--payload-drop-mode {tail,random,none}` (default `none`) and
`--payload-drop-pct 0..100` are also settable via the `ODP_PAYLOAD_DROP_MODE`
/ `ODP_PAYLOAD_DROP_PCT` environment variables (env wins if both set), which
is how the sweep driver below pins each child process to its own point.

**Sweep driver** — `aitestbed/tools/run_odp_payload_drop_sweep.py` automates
the full (mode, pct) grid: for every point it starts a *fresh* server (own
report file + recon dir, labelled `payloaddrop_<mode>_<pct>` so points never
collide), runs the whole-image client once against it, gracefully stops the
server (SIGINT, so its `finally:` block flushes partial images and writes
the report before exit -- a hard kill would lose it), and appends a row
(client + server metrics) to a combined summary JSON:

```bash
python aitestbed/tools/run_odp_payload_drop_sweep.py \
    --image-dir aicodec/aicodec/dataset/MMBench/MMBench_DEV_EN \
    --max-images 20 --modes tail,random \
    --pcts 10,20,30,40,50,60,70,80,90 \
    --report-dir results/odp_payload_drop_sweep
```

A `mode=none pct=0` baseline point is run first automatically (skip with
`--skip-baseline`). Results land in
`results/odp_payload_drop_sweep/payload_drop_sweep_summary.json`: one row
per (mode, pct) with `images_ok`/`images_total` from the client and
`token_delivery_rate` / `images_complete` / `images_partial` /
`tokens_recovered_total` / `tokens_expected_total` pulled straight from the
matching server report. Per-point server stdout/stderr logs
(`server_<label>.log`) and reconstructed PNGs
(`recon/<label>/`, skip with `--no-dump-recon`) are kept alongside for
debugging a specific point.

## Pipeline execution flow (`run_odp_mmbench_pipeline.py`)

Each profile in `--profiles` runs through the same 4 phases, independently
(no shared server/network state between profiles):

1. **Start the server** — launches `odp_image_udp_server.py --fresh
   --network-profile <tag>` and blocks until its log prints
   `"ODP image UDP server on"` (decoder warmed up); timeout via
   `--server-ready-timeout`.
2. **Run the orchestrator scenario** — writes a temp copy of
   `scenarios.yaml` with the current profile / payload-drop knobs baked in,
   then runs `odp_image_udp_whole` or `odp_image_udp_chunked` through the
   orchestrator, applying (or not applying) netem on `--interface`. After
   sending, waits `--post-send-settle` seconds for trailing datagrams, then
   waits for the server log to print `"session-complete control received"`.
3. **Verify the reconstruction** — checks that `recon/` has > 0 PNGs and the
   eval manifest has > 0 rows; aborts loudly instead of silently producing
   an empty result.
4. **Score with Qwen3-VL** — calls `mmbench_eval.py` via `conda run -n
   <eval-conda-env>` over `recon/`, producing `eval/summary.json`.
   `--skip-eval` skips this step if you only care about link/reconstruction
   metrics.

## Output layout and summary fields

```
<out-dir>/<profile_tag>/
    recon/            # server-reconstructed images + eval manifest.jsonl
    reports/           # server-side link report (*.json, timestamped)
    eval/              # mmbench_eval.py output: summary.json + per-item predictions
    logs/              # server.log / orchestrator.log / eval.log
    pipeline_summary.json   # per-profile summary (fields below)
<out-dir>/combined_summary.json   # array of every profile's pipeline_summary.json
```

`pipeline_summary.json` (also one array entry of `combined_summary.json`):

| Field | Meaning |
| --- | --- |
| `profile` | network profile tag for this run |
| `recon_dir` | path to the reconstructed-image folder fed to Qwen3-VL |
| `recon_png` / `recon_manifest_rows` | sanity-check counts from phase 3 |
| `link_report` | parsed copy of the server's `report.json` (see below) |
| `eval` | parsed copy of `mmbench_eval.py`'s `summary.json` (see below) |

`link_report` (written by `odp_image_udp_server.py`'s `save_report()`):

| Field | Meaning |
| --- | --- |
| `datagrams_received` / `datagrams_corrupt` | raw UDP receive counters |
| `images_received` / `images_complete` / `images_partial` | frames flushed to the decoder, fully intact vs. with holes/short tail |
| `chunks_expected_total` / `chunks_lost_total` | chunked-mode datagram accounting |
| `tokens_expected_total` / `tokens_recovered_total` | token-level accounting (works for both whole and chunked) |
| `datagram_loss_rate` | `chunks_lost_total / chunks_expected_total` |
| `token_delivery_rate` | `tokens_recovered_total / tokens_expected_total` — the main "how much survived" number |
| `partial_image_rate` | `images_partial / images_received` |

`eval` (written by `mmbench_eval.py`):

| Field | Meaning |
| --- | --- |
| `raw_accuracy`, `correct`, `total` | overall MMBench MCQ accuracy |
| `invalid_predictions` | answers that couldn't be parsed into a choice |
| `filtered_accuracy` | accuracy over a filtered subset (e.g. excluding invalid predictions) |
| `run_config` | echoes the eval CLI args used (model, limit, tensor-parallel-size, ...) |

## Troubleshooting

- **Server never prints the ready token / `--server-ready-timeout` fires**:
  cold-start weight loading can take a few minutes on a slow disk or first
  CUDA init; bump `--server-ready-timeout`, or pre-warm by running the
  server once manually.
- **`recon_png == 0` / "no reconstructed png" abort**: usually means every
  datagram for every image was lost (loss_pct too high for the chosen
  image count) or the server crashed mid-run — check `logs/server.log`.
- **Qwen eval fails with a conda error**: confirm `--eval-conda-env`
  actually exists (`conda env list`) and has `vllm` + the eval script's
  deps installed; `logs/eval.log` has the full traceback.
- **`--payload-drop-mode`/`--payload-drop-pct` seem to have no effect**:
  they only apply under the `lossy_payload` profile and `whole` mode by
  design (see "Two ways to lose tokens" above) — double check `--profiles`
  includes `lossy_payload` and `--mode whole` (or `--scenario
  odp_image_udp_whole`).
- **Want to compare runs quickly**: `combined_summary.json` has one entry
  per profile with `recon_png`, `recon_manifest_rows`, `link_report`, and
  `eval.raw_accuracy` already flattened for spreadsheet import.

