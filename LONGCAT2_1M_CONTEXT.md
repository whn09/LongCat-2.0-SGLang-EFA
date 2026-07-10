# LongCat-2.0-FP8 —— 1M(1,048,576 tokens)上下文支持方法与结果

**日期**:2026-07-10　**硬件**:4× p5en.48xlarge(8×H200 + 16×200Gbps EFA = 32 GPU),us-east-2
**模型**:meituan-longcat/LongCat-2.0-FP8(1.6T MoE,~48B 激活,FP8,LongCat Sparse Attention/NSA)

## 结论:1M 上下文可跑通 ✅

在 4 节点 32 GPU 上,单请求 **1,048,576 tokens prefill 成功完成**。关键是三个配置组合缺一不可。

---

## 一、支持 1M 的关键配置(缺一不可)

| 配置项 | 值 | 为什么必须 |
|---|---|---|
| **并行** | `--tp 32 --ep 32`(4 节点,无 CP) | ep32=tp32 → expert 完全 EP 分片,权重 1.6T/32 ≈ 50GB/卡(留 ~90GB 给 KV)。tp8/tp16 装不下或 KV 不够 |
| **KV cache dtype** | `--kv-cache-dtype fp8_e4m3` | fp8 让 KV 每 token 占用减半 → KV pool 达 **1.23M token**(bf16 只到 ~752K,装不下 1M)|
| **attention kernel** | `--nsa-prefill-backend flashmla_kv` | **fa3 内核不支持 fp8 KV**(报 `query and key must have the same dtype`);flashmla_kv(FlashMLA,fp8-native)兼容 bf16 query + fp8 KV |
| context 上限 | `--context-length 1048576` + env `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` | LongCat config 原生上限 262144(256K),1M 需 YARN 外推 override |
| mem-fraction | `--mem-fraction-static 0.88` | 0.88 下 KV pool 1.23M 够 1M;**不要用 0.93**(部分节点权重加载 OOM,world 残缺)|
| 单请求 | `--max-running-requests 1` | 释放全部 KV 给单个长请求 |

**CP(context parallel)不能用**:sglang 断言 `Context parallel only supports single machine
(tp_size<=8)`,tp32 跨机必须关 CP。所幸 tp32 已把 KV 按 32 切分,无需 CP 也能装下 1M。

## 二、启动命令

镜像:`ucclep-longcat2-efa149-v3`(UCCL-EP MoE over EFA)。脚本:`scripts/serve-tp32-1m.sh`,4 节点各跑一条(head = P5EN-1 内网 IP):

```bash
TP=32 EP=32 CP=0 NNODES=4 CTX_LEN=1048576 CHUNK=8192 MEM_FRAC=0.88 \
MAX_RUNNING=1 KV_DTYPE=fp8_e4m3 DSA_BACKEND=flashmla_kv \
bash serve-tp32-1m.sh <node-rank 0..3> <head-ip>
```

## 三、性能结果(单请求 1M prefill,UCCL-EP)

| chunked-prefill-size | 平均输入吞吐 | 1M 总耗时(约) | low-smem fallback |
|---|---|---|---|
| 16384 | 1401 tok/s | ~12.5 min | ❌ 触发(拖慢)|
| **8192** | **1974 tok/s** | **~8.9 min** | ✅ 无 |

- **chunk=8192 比 16384 快 ~40%**。原因:flashmla_kv 在 chunk=16384 时超 GPU 共享内存上限
  (`batch_size=16384 requires 327684B shared memory (max=232448B), using low-smem fallback kernel`),
  退回低速内核;chunk=8192 避开该 fallback,单块峰值吞吐可达 ~4900 tok/s。
- 全程平均吞吐仍随上下文增长下降(attention 的 O(n²) 特性:后段 chunk 要对已累积的百万级 KV 做注意力),
  例如 chunk=8192 从峰值 ~4900 逐步降到尾段 ~1200 tok/s。

## 四、对照:≤512K 用 bf16/fa3 更快(不需要 fp8/flashmla)

512K 及以下,bf16 KV + fa3 内核即可,且每 token 更快(无 fp8 转换、无 flashmla fallback):
- ISL=256K:TTFT 51s,5144 tok/s
- ISL=512K:TTFT 133s,3927 tok/s

**选型**:≤512K 用 bf16 + fa3(更快);>512K 到 ~1.2M 用 fp8 + flashmla_kv(唯一能装下)。

## 五、踩坑记录

1. **fa3 + fp8 KV 崩溃**:`RuntimeError: query and key must have the same dtype`
   (dsa_backend.py:2214)。必须换 flashmla_kv。
2. **mem0.93 + fp8 权重加载 OOM**:部分节点(131.8GB/卡)在 create_weights 阶段 OOM,world 残缺
   (head 挂在 100% 等掉线 rank),得到的"1M 结果"无效。用 mem≤0.88 让 4 节点都活。
3. **tp8 完全不可行**:sglang 把 MoE EP 嵌套在 TP world 内(moe_ep_size ≤ tp_size),tp8 时每卡权重
   135GB 撑爆 H200,PP/EP/CP/cpu-offload 都救不了。必须 tp≥16;1M 用 tp32。
4. **1M 单请求 KV 上限 = max_total_num_tokens**(fp8/mem0.88 下 1.23M)。更长需更多节点或更省 KV。

## 六、UCCL-EP vs DeepEP(1M 上下文,同 tp32/ep32/fp8/flashmla_kv/chunk8192)

| MoE 后端 | 镜像 | 1M 输入吞吐 | 备注 |
|---|---|---|---|
| **UCCL-EP** | ucclep-longcat2-efa149-v3 | **1974 tok/s** | |
| **DeepEP** | deepep-longcat2-efa149-v3 | **1972 tok/s** | 与 UCCL 几乎一致(差 <0.1%)|

**结论:1M 超大上下文下,UCCL-EP 与 DeepEP 吞吐基本无差异。** 原因:1M prefill 的瓶颈是
**attention 的 O(n²) 计算**(后段 chunk 要对已累积的百万级 KV 做注意力),MoE dispatch/combine
只占总时间一小部分,因此两种 EP 后端的差异被 attention 主导、几乎不可见。
(对照:在**小消息 decode 场景**下 UCCL-EP 明显快于 DeepEP —— 见 LOWLATENCY_symm_vs_deepep_vs_uccl.md;
但那是 MoE all-to-all 主导的场景,与 1M prefill 的 attention 主导场景不同。)

## 七、原始日志

- UCCL chunk16384:`/opt/dlami/nvme/tp32_1m_flashmla.log`(1401 tok/s,触发 low-smem fallback)
- UCCL chunk8192:`/opt/dlami/nvme/tp32_1m_chunk8192.log`(1974 tok/s)
- DeepEP chunk8192:`/opt/dlami/nvme/tp32_1m_deepep.log`(1972 tok/s)
