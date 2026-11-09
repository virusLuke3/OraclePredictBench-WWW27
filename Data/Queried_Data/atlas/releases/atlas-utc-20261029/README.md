---
license: cc-by-4.0
language:
- en
configs:
- config_name: uma
  data_files:
  - split: train
    path: atlas/releases/atlas-utc-20261029/data/native/polygon_uma_request_rounds.parquet
- config_name: chainlink
  data_files:
  - split: train
    path: atlas/releases/atlas-utc-20261029/data/native/chainlink_staking_v02_events.parquet
- config_name: tellor
  data_files:
  - split: train
    path: atlas/releases/atlas-utc-20261029/data/native/tellor_disputes.parquet
- config_name: flare_ftsov2
  data_files:
  - split: train
    path: atlas/releases/atlas-utc-20261029/data/native/flare_reward_epochs.parquet
- config_name: pyth
  data_files:
  - split: train
    path: atlas/releases/atlas-utc-20261029/data/native/pyth_ois_reward_epochs.parquet
- config_name: chronicle
  data_files:
  - split: train
    path: atlas/releases/atlas-utc-20261029/data/native/chronicle_ethereum_events.parquet
- config_name: redstone_push
  data_files:
  - split: train
    path: atlas/releases/atlas-utc-20261029/data/native/redstone_ethereum_push_events.parquet
- config_name: dia
  data_files:
  - split: train
    path: atlas/releases/atlas-utc-20261029/data/native/dia_staking_withdrawals.parquet
- config_name: uma_prediction
  data_files:
  - split: train
    path: atlas/releases/atlas-utc-20261029/data/prediction/uma_prediction_tasks.parquet
- config_name: daily_measurements
  data_files:
  - split: train
    path: atlas/releases/atlas-utc-20261029/data/daily/daily_measurements.parquet
pretty_name: 'Oracle4SD Atlas: eight oracle branches, UMA tasks and daily measurements'
tags:
- blockchain
- oracles
- economic-measurement
- time-series
---
# Oracle4SD Atlas

This release contains ten logical data products for studying oracle participation, incentives and accountability. Eight products retain protocol-native observations; the other two provide UMA request-level prediction tasks and daily measurements. The full release has **100 catalogued analysis tables**. Tables shared across products are counted once in that inventory. Additional Parquet files hold supporting metadata, mirrored nested records or retained reproduction inputs; they are listed separately in the package inventory.

**Version:** `atlas-utc-20261029`. [Processing code](https://github.com/Oracle4CEG/Oracle4SD-Atlas/tree/71c0e19d3ae2b47d763243371f1bfb97750db61d) · [Manuscript](https://github.com/Oracle4CEG/Oracle4SD-Atlas/blob/9d7a227d4895f31a5c2b865d4dd66e3d1ce88b8a/paper/working_revision/main.pdf), [full table catalogue](https://huggingface.co/datasets/Oracle4CEG/OracleEconLab-Oracle-Incentives-v1/blob/main/atlas/releases/atlas-utc-20261029/metadata/table_catalog.json), [field dictionary](https://huggingface.co/datasets/Oracle4CEG/OracleEconLab-Oracle-Incentives-v1/blob/main/atlas/releases/atlas-utc-20261029/metadata/field_dictionary.csv), [product membership](https://huggingface.co/datasets/Oracle4CEG/OracleEconLab-Oracle-Incentives-v1/blob/main/atlas/releases/atlas-utc-20261029/metadata/logical_products.json).

## Ten datasets

Each Viewer configuration opens one representative table with a consistent schema. The table count below includes all native and shared tables associated with that product; it is not the number of tables loaded by the Viewer. Use the product manifest to retrieve additional tables. A `train` split is only a loading label here and does not define a modelling partition.

| Configuration | Product | Preview rows | Associated tables |
| --- | --- | ---: | ---: |
| `uma` | UMA | 42,676 | 42 |
| `chainlink` | Chainlink | 174,890 | 21 |
| `tellor` | Tellor | 13 | 32 |
| `flare_ftsov2` | Flare FTSOv2 | 183 | 29 |
| `pyth` | Pyth | 49 | 24 |
| `chronicle` | Chronicle | 140,526 | 19 |
| `redstone_push` | RedStone Push | 614,319 | 18 |
| `dia` | DIA | 129 | 19 |
| `uma_prediction` | UMA prediction | 810 | 3 |
| `daily_measurements` | Daily measurements | 13,269 | 12 |

UMA, Chainlink, Tellor, Flare FTSOv2 and Pyth are the detailed panels. Chronicle, RedStone Push and DIA are narrower extensions. The Push sample does not cover all RedStone services or AVS economics. Coverage varies by implementation, chain, period and source; first and last observations do not establish uninterrupted collection.

## Loading data

```python
from datasets import load_dataset

dataset = load_dataset(
    "Oracle4CEG/OracleEconLab-Oracle-Incentives-v1",
    "daily_measurements",
    revision="atlas-utc-20261029",
    split="train",
)
```

For tables beyond a configuration's preview, download the declared file directly. This avoids combining tables that have different observation units or schemas.

```python
from huggingface_hub import hf_hub_download
import pandas as pd

path = hf_hub_download(
    repo_id="Oracle4CEG/OracleEconLab-Oracle-Incentives-v1", repo_type="dataset",
    revision="atlas-utc-20261029",
    filename="atlas/releases/atlas-utc-20261029/data/prediction/feature_availability.parquet",
)
availability = pd.read_parquet(path)
```

The repository's `scripts/load_atlas_product.py` lists product members, downloads selected tables and verifies their SHA-256 hashes. Downloading every physical table requires about 25.5 GB, plus space for caches or analysis outputs.

## Measurement and reuse

Daily measurements contain **13,269 rows** grouped by UTC date, protocol branch, mechanism, role, asset and economic stage. They retain both the source population and the subset with supported times. All rows carry a partial-coverage qualification. Missing days are not filled with zero. Escrow exposure covers observed closed requests; signing weights describe epoch policies and are not deposited capital.

The UMA task contains **810 requests and 16 candidate features**. Its cohort is retrospectively restricted to challenged, linked, settled and transfer-reconciled requests. The label compares the protocol's proposed and resolved integers, including special resolution values; it is not independent factual truth. Four candidate features—bond, configured reward, reward/bond ratio and adapter version—still require historical evidence that binds their values to proposal time. Exclude them from a claim of verified decision-time prediction until that evidence is supplied. Feature-level availability and missingness are published with the task. No model-performance claim is made for this release.

Native tables preserve source fields and add supported UTC clocks. Common tables retain their earlier fields; join their clock companions by the declared record identifier before selecting time windows. Epoch intervals, snapshots, accruals, claims, payments and applied losses have different meanings. Shared core tables contain multiple branches and overlapping representations: filter the intended branch and choose one observation layer before aggregation.

Automated checks cover file identity, physical schemas, temporal transformations and specified accounting rules. Human review of historical bindings and an independent reader's Google-hosted Colab run remain pending. Historical acquisition receipts, some block times and the complete cross-chain relay are incomplete. Addresses do not establish independent organizations. See the evidence records for each limitation rather than treating unknown values as zero.

## Versions and citation

The previous 15-config experimental release remains available at [its fixed revision](https://huggingface.co/datasets/Oracle4CEG/OracleEconLab-Oracle-Incentives-v1/tree/356fb1aeac18fb4e6672ca1ea56988dac72c3417). Its files are retained. This release changes the default Viewer to ten products and adds a separate version directory; it does not relabel earlier model outputs as newly validated data.

Cite: Oracle4CEG. *Oracle4SD Atlas: eight oracle branches, UMA tasks and daily measurements*. Version `atlas-utc-20261029`. Hugging Face, 2026. Record the resolved commit hash with each analysis. No DOI has been assigned to this release.

Project-created data and metadata use CC BY 4.0. Third-party source rights remain as recorded in [DATA_LICENSE.md](https://huggingface.co/datasets/Oracle4CEG/OracleEconLab-Oracle-Incentives-v1/blob/main/atlas/releases/atlas-utc-20261029/DATA_LICENSE.md). Code licensing is specified in the code repository.
