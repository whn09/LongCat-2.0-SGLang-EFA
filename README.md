# LongCat-2.0-SGLang-EFA

在 4× p5en.48xlarge（8×H200 + 16×200Gbps EFA）上，用 **SGLang + UCCL-EP（MoE all-to-all over EFA）
+ Mooncake（KV over EFA）** 部署 **meituan-longcat/LongCat-2.0-FP8**（1.6T MoE，~48B 激活，FP8）的
PD 分离（Prefill/Decode disaggregation）方案。

## 组成

| 文件 | 说明 |
|---|---|
| `Dockerfile.sglang-ucclep` | 镜像：nightly-cu13 + EFA 1.49 + GDRCopy + Mooncake + UCCL-EP（DeepEP 兼容 `deep_ep` wrapper，原生 EFA，无 NVSHMEM）+ `NCCL_NET_PLUGIN` 修复 |
| `patch_uccl_ep_empty_tensor.sh` | 构建时补丁：UCCL-EP empty-tensor |
| `patch_longcat2_disagg_ngram.sh` | 构建时补丁：把 n-gram embedding 接入 PD 分离调度循环 |
| `scripts/serve-pd.sh` | 启动 Prefill / Decode 实例（tp16/ep16，deepep MoE over EFA） |
| `scripts/serve-router.sh` | 启动 PD router |

> 注：`--moe-a2a-backend deepep` 是 UCCL-EP 在 sglang 里的接口开关（UCCL-EP 提供 DeepEP 兼容
> `deep_ep`），底层实际走 UCCL-EP 原生 EFA，**不是** NVSHMEM DeepEP。

## 快速开始

```bash
# 1. 构建镜像（context = repo 根，patch 脚本会被 COPY）
docker build -f Dockerfile.sglang-ucclep -t ucclep-longcat2-efa149-v3 .

# 2. 启动 PD 分离（每节点一条命令）
#    Prefill = 2 节点，Decode = 2 节点（各 tp16/ep16，模型 1.6T 需 16 GPU）
bash scripts/serve-pd.sh prefill 0 <prefill_head_ip>   # prefill head
bash scripts/serve-pd.sh prefill 1 <prefill_head_ip>   # prefill node1
bash scripts/serve-pd.sh decode  0 <decode_head_ip>    # decode head
bash scripts/serve-pd.sh decode  1 <decode_head_ip>    # decode node1

# 3. 在 prefill head 启动 router
bash scripts/serve-router.sh <prefill_ip> <decode_ip>
```

关键 env / 参数（serve-pd.sh 已内置）：`MOONCAKE_PROTOCOL=efa`、`FI_HMEM_CUDA_USE_DMABUF=0`
（绕过 EFA 1.49 dmabuf 在 cuda:1 的注册失败）、`--chunked-prefill-size 16384 --mem-fraction-static 0.85`、
`--nsa-prefill-backend fa3`、`--kv-cache-dtype bfloat16`、`--moe-a2a-backend deepep`、`--disable-radix-cache`。

---

# Benchmark（EFA 1.49 + chunk=16384）

**测试日期**：2026-07-09　**模型**：meituan-longcat/LongCat-2.0-FP8（1.6T MoE，~48B 激活，FP8）
**硬件**：4× p5en.48xlarge（8×H200 + 16×200Gbps EFA），us-east-2

## 部署拓扑（1P1D）

| 角色 | 节点 | 配置 |
|---|---|---|
| **Prefill 实例** | P5EN-1(head) + P5EN-2 | tp16 / ep16 / nnodes2，deepep `normal` 模式 |
| **Decode 实例** | P5EN-3(head) + P5EN-4 | tp16 / ep16 / nnodes2，deepep `low_latency` 模式 |
| **Router** | P5EN-1 | round_robin，PD-disaggregation |

- **镜像**：`ucclep-longcat2-efa149-v3`（**EFA installer 1.49** + `NCCL_NET_PLUGIN` 修复 baked-in，
  避免 NCCL→TCP 回退）。
- **KV 传输**：Mooncake over EFA（`MOONCAKE_PROTOCOL=efa`，16 NIC 注册），
  `FI_HMEM_CUDA_USE_DMABUF=0`（绕过 1.49 dmabuf 在 cuda:1 的注册失败）。
- **关键参数**：`--chunked-prefill-size 16384`、`--mem-fraction-static 0.85`
  （chunk16384 在 mf0.92 会 OOM，必须降到 0.85）、`--nsa-prefill-backend fa3`、
  `--kv-cache-dtype bfloat16`、`--moe-a2a-backend deepep`、`--disable-radix-cache`、
  `--max-running-requests 64`。
- **Attention backend**：dsa（LongCat Sparse Attention 自动选择）。

## 压测方法

`sglang.bench_serving`，`dataset=random`，`--random-range-ratio 1.0`（严格 ISL/OSL，长度精确）；
此 build **默认忽略 EOS**（保证跑满 OSL）；`--warmup-requests <conc>`、`--flush-cache`；
temperature=0。经 router（:8000）端到端。

---

## 一、Prefill —— TTFT（chunked-prefill-size = 16384，OSL = 1）

### ISL = 1024

| conc | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Input tput (tok/s) |
|-----:|---------------:|-----------------:|--------------:|-------------------:|
|    1 |         232.05 |           231.66 |        236.45 |            4403 |
|    2 |         437.15 |           426.75 |        503.48 |            4672 |
|    4 |         665.08 |           660.66 |        807.75 |            6122 |
|    8 |         974.41 |           982.39 |       1130.96 |            8329 |
|   16 |        1636.71 |          1610.18 |       1830.12 |            9930 |
|   32 |        3192.64 |          3109.28 |       3594.92 |           10210 |
|   64 |        5892.90 |          5617.13 |       9920.11 |           10730 |

### ISL = 8192

| conc | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Input tput (tok/s) |
|-----:|---------------:|-----------------:|--------------:|-------------------:|
|    1 |        1088.59 |          1061.48 |       1516.68 |            7521 |
|    2 |        2016.19 |          2029.88 |       2669.57 |            8063 |
|    4 |        3947.81 |          3979.64 |       4648.65 |            8256 |
|    8 |        7635.97 |          7801.32 |      11775.72 |            8346 |
|   16 |       14666.68 |         15609.51 |      19498.95 |            8402 |
|   32 |       28513.00 |         31297.40 |      34204.76 |            8388 |
|   64 |       55983.92 |         62562.46 |      64350.20 |            8374 |

## 二、Decode —— TPOT（ISL = 1024，OSL = 1024）

| conc | Mean TPOT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Output tput (tok/s) |
|-----:|---------------:|-----------------:|--------------:|--------------------:|
|    1 |          24.85 |            24.85 |         24.88 |               39.73 |
|    2 |          26.23 |            26.44 |         26.49 |               66.70 |
|    4 |          27.26 |            27.25 |         27.52 |              138.91 |
|    8 |          28.92 |            28.95 |         29.33 |              214.30 |
|   16 |          35.60 |            35.76 |         35.87 |              384.24 |
|   32 |          38.51 |            38.57 |         40.68 |              635.67 |
|   64 |          40.23 |            39.92 |         59.08 |              666.32 |

---

## 三、观察与结论

1. **Prefill TTFT 随并发近似线性增长**（单 prefill 实例，FCFS 调度）。
   - ISL1024：c1=232ms → c64=5893ms；输入吞吐在 c16+ 饱和到 ~10.7k tok/s @c64。
   - ISL8192：c1=1089ms → c64=55984ms；输入吞吐稳定 ~8.4k tok/s。
2. **chunk=16384 vs 2048**：对照旧 chunk2048 数据（ISL1024 c1=225/c16=1971/c64=6987ms），
   **chunk16384 在中高并发更优**：c16 从 1971→1637ms（↓17%），c64 从 6987→5893ms（↓16%）。
   代价是必须把 mem-fraction 从 0.92 降到 0.85（KV pool 缩小），prefill 实例 KV 需求小可接受。
3. **Decode TPOT 稳定在 25–40 ms/token**（c1→c64），聚合输出吞吐 c32≈636、c64≈666 tok/s。
   得益于 deepep `low_latency` 模式保留 decode CUDA graph。
   - c64 P99 TPOT 抖到 59ms（尾延迟），Mean 仍 40ms，说明高并发下有少量长尾。
4. **EFA 1.49 + NCCL_NET_PLUGIN 修复后**：prefill/decode 全程稳定，无 NCCL→TCP 回退，
   无 Mooncake KV-over-EFA 报错（16 NIC 全注册成功）。

> 说明：本 benchmark 使用未含 Zero-Expert 优化的 UCCL-EP；LongCat-2.0 有 128 个 zero-computation
> expert，客户生产版对 dispatch 做了 Zero-Expert 优化（路由到 zero-expert 的 token 跳过跨机
> all-to-all），实际 decode 会更快。
