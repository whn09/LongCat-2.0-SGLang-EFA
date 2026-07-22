# LongCat-2.0-SGLang-EFA

在 4× p5en.48xlarge（8×H200 + 16×200Gbps EFA）上，用 **SGLang + UCCL-EP（MoE all-to-all over EFA）+ Mooncake（KV over EFA）** 部署 **meituan-longcat/LongCat-2.0-FP8**（1.6T MoE，~48B 激活，FP8）的 PD 分离（Prefill/Decode disaggregation）方案。

## 组成

| 文件 | 说明 |
|---|---|
| `Dockerfile.sglang-ucclep` | 镜像：nightly-cu13-20260715 + EFA 1.49 + GDRCopy + Mooncake + UCCL-EP（DeepEP 兼容 `deep_ep` wrapper，原生 EFA，无 NVSHMEM）+ `NCCL_NET_PLUGIN` 修复。**所有 LongCat 真-EP 修复以上游 PR diff 的形式在构建时 `git apply`**（见 `patches/`），不再用临时 sed 补丁 |
| `patches/*.diff` | 上游 PR 的标准 diff（构建时 `git apply`）：sglang #30975/#31311/#31312/#31134 + UCCL #1020/#1021。每个都可追溯到一个 PR，PR 合并进 base 后即可删 |
| `scripts/serve-pd.sh` | 启动 Prefill / Decode 实例（tp16/ep16，PD 分离；prefill=deepep normal，decode=deepep low_latency） |
| `scripts/serve-tp16.sh` | 启动**单个非分离**实例（tp16/ep16 跨 2 节点）——只有 2 节点、无法跑 PD 时用它做正确性/并发测试；含 `smoke` 子命令 |
| `scripts/serve-tp32-1m.sh` | 启动**单个非分离**实例，tp32/ep32 跨 **4 节点**，面向 **1M 长上下文**（专家全 EP 分片省显存 + `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` 覆盖 256K 上限）；见 `LONGCAT2_1M_CONTEXT.md` |
| `scripts/serve-router.sh` | 启动 PD router |

> 注：`--moe-a2a-backend deepep` 是 UCCL-EP 在 sglang 里的接口开关（UCCL-EP 提供 DeepEP 兼容
> `deep_ep`），底层实际走 UCCL-EP 原生 EFA，**不是** NVSHMEM DeepEP。

### 内置的上游修复（构建时 `git apply patches/*.diff`）

让 LongCat-2.0 在真 EP（`--moe-a2a-backend deepep`）下**跑通且输出正确**所需的全部上游 PR：

| PR | 修什么 | 状态 | 本仓如何应用 |
|---|---|---|---|
| sglang [#30975](https://github.com/sgl-project/sglang/pull/30975) | scheduler moe-topk gate（否则 `--moe-a2a-backend` 对 LongCat 静默失效） | **已合并** | 仍保留 diff（base nightly-20260715 早于合并点）；base 升过合并点后即可删 |
| sglang [#31311](https://github.com/sgl-project/sglang/pull/31311) | MoE 后与 DeepEPMoE combine 的双重 all_reduce（乱码）+ ScMoE dense 分支 gather（RoPE 崩） | **已合并** | 仍保留 diff（base nightly-20260715 早于合并点）；base 升过合并点后即可删 |
| sglang [#31312](https://github.com/sgl-project/sglang/pull/31312) | n-gram token-table 在 padded（cuda-graph）decode batch 下越界崩 | **已合并** | 仍保留 diff（base nightly-20260715 早于合并点）；base 升过合并点后即可删 |
| sglang [#31134](https://github.com/sgl-project/sglang/pull/31134) | 把 n-gram 准备接入 PD 分离调度循环 | **已合并** | 仍保留 diff（base nightly-20260715 早于合并点）；base 升过合并点后即可删 |
| UCCL [#1020](https://github.com/uccl-project/uccl/pull/1020) | internode TMA sender buffer 16384→20480（hidden=8192） | open | `patches/uccl-pr-1020-tma-hidden8192.diff` |
| UCCL [#1021](https://github.com/uccl-project/uccl/pull/1021) | WriteImm expert_idx 9→10 bit（768 expert，否则 LL 启动崩） | **已合并** | 仍保留 diff（`UCCL_REF` 钉在合并点之前）；`UCCL_REF` 升过合并点后即可删 |
| UCCL [#1016](https://github.com/uccl-project/uccl/pull/1016) | kNumMaxTopK 9→16（LongCat moe_topk=12） | **已合并** | 无需（已在 uccl main） |
| UCCL empty-tensor（topk_idx_ptr==0 / num_tokens==0） | DP-attention 空 batch 不崩 | **已在 main** | 无需 |

> `git apply` 依赖上下文匹配，故 base 镜像 tag **钉死日期**（`nightly-dev-cu13-20260715`）。
> 上述 open 的 PR 一旦合并进某个 nightly，把 base tag 升上去、删掉对应 diff 即可（sglang #30387
> zero-expert 修复和 #31154 ngram 重构就是这样被 20260715 base 自带、故本仓无需再 patch）。

## 快速开始

```bash
# 1. 构建镜像（context = repo 根，patches/ 会被 COPY 并 git apply）
docker build -f Dockerfile.sglang-ucclep -t ucclep-sglang-efa:latest .

# 2a. PD 分离（4 节点：Prefill 2 + Decode 2，各 tp16/ep16，模型 1.6T 需 16 GPU）
bash scripts/serve-pd.sh prefill 0 <prefill_head_ip>   # prefill head
bash scripts/serve-pd.sh prefill 1 <prefill_head_ip>   # prefill node1
bash scripts/serve-pd.sh decode  0 <decode_head_ip>    # decode head
bash scripts/serve-pd.sh decode  1 <decode_head_ip>    # decode node1
bash scripts/serve-router.sh <prefill_ip> <decode_ip>  # 在 prefill head

# 2b. 只有 2 节点时：单个非分离实例（跑正确性/并发测试）
bash scripts/serve-tp16.sh 0 <head_ip>   # head
bash scripts/serve-tp16.sh 1 <head_ip>   # node1
bash scripts/serve-tp16.sh smoke         # 就绪后在 head 上

# 2c. 1M 长上下文（4 节点 tp32/ep32 单实例，非分离）
bash scripts/serve-tp32-1m.sh 0 <head_ip>   # rank 0 (head) .. rank 3
#   默认即 1M：fp8 KV（KV_DTYPE=fp8_e4m3）撑大 KV 池到 ~1.23M token + flashmla_kv 内核。
#   （bf16 只到 ~752K 装不下 1M；fa3 在 fp8 下崩 "q/k must have same dtype"，故 fp8 必配 flashmla_kv）
#   ≤512K 更快：显式 KV_DTYPE=bfloat16 DSA_BACKEND=fa3 CTX_LEN=524288 bash serve-tp32-1m.sh ...
```

关键 env / 参数（serve-pd.sh 已内置）：`MOONCAKE_PROTOCOL=efa`、`FI_HMEM_CUDA_USE_DMABUF=0`
（绕过 EFA 1.49 dmabuf 在 cuda:1 的注册失败）、`--chunked-prefill-size 16384 --mem-fraction-static 0.85`、
`--nsa-prefill-backend fa3`、`--kv-cache-dtype bfloat16`、`--moe-a2a-backend deepep`、`--disable-radix-cache`。

---

# Benchmark（EFA 1.49 + chunk=16384）

**测试日期**：2026-07-16　**模型**：meituan-longcat/LongCat-2.0-FP8（1.6T MoE，~48B 激活，FP8）
**硬件**：4× p5en.48xlarge（8×H200 + 16×200Gbps EFA），us-east-2

## 部署拓扑（1P1D）

| 角色 | 节点 | 配置 |
|---|---|---|
| **Prefill 实例** | P5EN-1(head) + P5EN-2 | tp16 / ep16 / nnodes2，deepep `normal` 模式 |
| **Decode 实例** | P5EN-3(head) + P5EN-4 | tp16 / ep16 / nnodes2，deepep `low_latency` 模式 |
| **Router** | P5EN-1 | round_robin，PD-disaggregation |

- **镜像**：`ucclep-sglang-efa:latest`（base nightly-cu13-20260715 + **EFA installer 1.49** +
  `NCCL_NET_PLUGIN` 修复 baked-in 避免 NCCL→TCP 回退 + `patches/*.diff` 的上游修复）。
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

## 一、Prefill —— TTFT（PD 分离后端对比，2026-07-16，delivery 镜像，4 节点 1P1D）

真正的交付形态：**PD 分离**，Prefill 实例 = 2 节点 tp16/ep16，Decode 实例 = 2 节点 tp16/ep16，
KV 经 Mooncake over EFA 传输，router 转发。端到端 TTFT（含 KV 传输）。对比两种 MoE 后端：
- **deepep**：prefill=`deepep normal`，decode=`deepep low_latency`（`serve-pd.sh` 默认）
- **none**：prefill/decode 都 `MOE_A2A=none`（EP-over-TP 基线）

bf16 KV，`sglang.bench_serving` random，`--random-range-ratio 1.0`（严格等长）。KV 池
`max_total_num_tokens=102656`，故 64K 只能测到 conc≈2、128K 输入无法测（超池，需 fp8 KV，
而 Mooncake 传 fp8 KV 暂不支持）。**粗体 = 更优。**

**ISL = 1024**

| conc | deepep（P normal）TTFT / tput | none TTFT / tput |
|-----:|------------------------------:|-----------------:|
|    1 |                       675 ms  |     **~640 ms**  |
|    4 |            743 ms / 5491      | **635 ms / 6415**|
|    8 |           1044 ms / 7808      | **938 ms / 8653**|
|   16 |       **1572 ms** / **10367**|    1633 ms / 9955|
|   32 |       **2899 ms** / **11211**|    3218 ms / 9954|
|   64 |       **4707 ms** / **13539**|    5856 ms / 10937|

**ISL = 8192**

| conc | deepep（P normal）TTFT / tput | none TTFT / tput |
|-----:|------------------------------:|-----------------:|
|    1 |        **985 ms** / —        |   1092 ms / 7497 |
|    4 |    **3466 ms** / **9129**    |   3986 ms / 8177 |
|    8 |    **6504 ms** / **9832**    |   7702 ms / 8277 |
|   16 |   **12395 ms** / **9929**    |  14842 ms / 8305 |

**ISL = 65536**（KV 池限制，conc≥2 已靠 chunked-prefill 排队饱和）

| conc | deepep（P normal）TTFT / tput | none TTFT / tput |
|-----:|------------------------------:|-----------------:|
|    1 |    **7594 ms** / **8629**    |   8823 ms / 7427 |
|    2 |   **14398 ms** / **8818**    |  16803 ms / 7556 |

- **crossover 点随 ISL 增大而左移**：ISL=1024 在 conc≈16 处 deepep 反超；ISL=8192 提前到 **conc=1**；
  ISL=65536 则 deepep 全程领先（吞吐 +15%）。
- 机理：低并发/短输入时真 EP 的 all-to-all 固定开销 > 省下的计算，`none` 略快；
  大 batch / 长序列时 all-to-all 被摊薄，且 deepep 省掉 none 的 all-gather 权重复制，`deepep normal` 反超。
- 选型：**短输入低并发用 `none`；长输入（≥8K）或高并发用 `deepep normal`**。

## 二、Decode —— TPOT（PD 分离后端对比，2026-07-16，delivery 镜像，4 节点 1P1D，ISL=64，OSL=1024）

同一 PD 拓扑（decode 实例 = 2 节点 tp16/ep16，独占 GPU 不与 prefill 争），对比 decode 后端：
- **deepep LL**：decode=`deepep low_latency`（cuda-graph 常开）。LL 参数 `MAXRUN=32 DDT=128`
  （默认 MAXRUN=128/DDT=256 会 cuda-graph capture OOM），并发上限 = MAXRUN=32。
- **none**：decode=`MOE_A2A=none`（EP-over-TP）。

| conc | deepep LL — TPOT | none — TPOT |
|-----:|-----------------:|------------:|
|    1 |          46.5 ms | **24.1 ms** |
|    4 |          47.5 ms | **26.9 ms** |
|    8 |          48.3 ms | **28.6 ms** |
|   16 |          48.3 ms | **35.0 ms** |
|   32 |          55.4 ms | **38.2 ms** |

- **2 机 16 卡的 decode 实例规模下，`none` 全程比 LL 快**，即便 LL 的 cuda-graph 生效、GPU 99%。
- 但 **LL 的 TPOT 曲线明显更平**：conc 1→32 只从 46.5 涨到 55.4 ms（+19%），而 none 从 24.1 涨到
  38.2 ms（+58%）。趋势上并发越高两者越接近。
- 机理：`none` decode 纯算力受限（TP all-gather 走节点内 NVLink）；LL 每 step 跨节点 EFA all-to-all，
  那 99% GPU 含等 RDMA 的忙等。**LL 的真正拐点需要更大并发**（超过单 decode 实例 MAXRUN=32 上限，
  需多 decode 实例横向扩），此规模下 decode 用 `none` 最优。

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

---

# 长上下文（1M tokens）—— tp32/ep32 单实例（`scripts/serve-tp32-1m.sh`）

在 4 节点 32 GPU 上单实例（非 PD 分离）跑通了**单请求 1,048,576 tokens（1M）prefill**。
详见 `LONGCAT2_1M_CONTEXT.md`。

## 支持 1M 的三个关键（缺一不可）

| 配置 | 值 | 为什么必须 |
|---|---|---|
| 并行 | `--tp 32 --ep 32`（4 节点，无 CP） | ep32=tp32 → expert 完全 EP 分片，权重 1.6T/32≈50GB/卡（tp8 会 OOM）|
| **KV dtype** | `--kv-cache-dtype fp8_e4m3` | fp8 让 KV 减半 → KV pool 达 1.23M token（bf16 只到 ~752K，装不下 1M）|
| **attention kernel** | `--nsa-prefill-backend flashmla_kv` | fa3 内核不支持 fp8 KV（报 `query and key must have the same dtype`）；flashmla_kv 兼容 |

外加：`--context-length 1048576` + env `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`（原生上限 256K，1M 需 YARN 外推）；
`--mem-fraction-static 0.88`（不要 0.93，会在权重加载 OOM）；CP 必须关（sglang 的 CP 仅支持 tp≤8）。

启动（4 节点各一条，head=节点0 内网 IP）：
```bash
TP=32 EP=32 CP=0 NNODES=4 CTX_LEN=1048576 CHUNK=8192 MEM_FRAC=0.88 \
MAX_RUNNING=1 KV_DTYPE=fp8_e4m3 DSA_BACKEND=flashmla_kv \
bash scripts/serve-tp32-1m.sh <node-rank 0..3> <head-ip>
```

## 1M prefill 性能（单请求）

| chunked-prefill-size | 输入吞吐 | 1M 总耗时（约） | 说明 |
|---|---|---|---|
| 16384 | 1401 tok/s | ~12.5 min | flashmla_kv 触发 low-smem fallback（拖慢）|
| **8192** | **1974 tok/s** | **~8.9 min** | 避开 fallback，+40% |

- 更短上下文（≤512K）用 bf16 KV + fa3 更快：256K TTFT 51s（5144 tok/s）、512K TTFT 133s（3927 tok/s）。
- **UCCL-EP vs DeepEP @1M**：1974 vs 1972 tok/s，**基本无差异** —— 1M prefill 瓶颈是 attention 的
  O(n²) 计算（百万级 KV），MoE all-to-all 只占小部分，EP 后端差异被淹没。
  （小消息 decode 场景 UCCL-EP 明显快于 DeepEP，是不同 regime。）
