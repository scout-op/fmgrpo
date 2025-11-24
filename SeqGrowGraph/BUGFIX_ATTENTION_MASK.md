# Bug Fix: Attention Mask Shape Mismatch in LPIM

## 问题描述

在 Stage I 训练时遇到如下错误：

```
RuntimeError: The shape of the 3D attn_mask is torch.Size([1600, 21, 21]), 
but should be (12800, 21, 21).
```

## 根本原因

`PriorEncoder` 中使用的 `attn_mask` 参数在 PyTorch 的 `MultiheadAttention` 中有特殊要求：

- 当使用 3D mask 时，第一维必须是 `batch_size * num_heads`
- 我们的实现中只提供了 `batch_size` 维度，导致形状不匹配

公式：`期望形状 = (B*M*num_heads, N+1, N+1)`
实际形状 = `(B*M, N+1, N+1)`

其中：
- B = batch_size (8)
- M = max_lanes (200 in actual data)
- num_heads = 8
- N = max_points_per_lane (20)

8 * 200 = 1600 (实际)
8 * 200 * 8 = 12800 (期望)

## 修复方案

将 `attn_mask` 改为 `key_padding_mask`：

### 修改前
```python
# 使用 attn_mask (3D)
mask = mask.unsqueeze(1).expand(-1, N + 1, -1)  # [B*M, N+1, N+1]
mask = mask.masked_fill(mask == 0, float('-inf'))
src2, _ = self.self_attn(src, src, src, attn_mask=mask)
```

### 修改后
```python
# 使用 key_padding_mask (2D)
key_padding_mask = torch.cat([query_mask, mask], dim=1)  # [B*M, N+1]
key_padding_mask = (key_padding_mask == 0)  # True = ignore
src2, _ = self.self_attn(src, src, src, key_padding_mask=key_padding_mask)
```

### 优势
1. **避免维度问题**: `key_padding_mask` 是 2D，不需要考虑 num_heads
2. **语义更清晰**: 专门用于标记 padding 位置
3. **性能更好**: 2D mask 比 3D mask 内存占用更小

## 修改的文件

- `/seq_grow_graph/lane_diffusion/lpim.py`:
  - `TransformerEncoderLayer.forward()`: 参数从 `src_mask` 改为 `src_key_padding_mask`
  - `PriorEncoder.forward()`: mask 构造逻辑更新

## 验证

运行测试脚本：
```bash
cd /data/roadnet_data/lanediff/mmdetection3d/projects/SeqGrowGraph
python test_lpim_fix.py
```

如果显示 "Fix verified!"，说明修复成功。

## 重新开始训练

修复后可以直接重新开始训练：

```bash
# 清理之前的输出（可选）
rm -rf work_dirs/seq_grow_graph_lanediffusion/

# 重新开始训练
./tools/dist_train.sh \
  configs/seq_grow_graph/seq_grow_graph_lanediffusion.py \
  8
```

## 技术细节

### key_padding_mask vs attn_mask

| 特性 | key_padding_mask | attn_mask |
|------|------------------|-----------|
| 维度 | 2D: `(N, S)` | 2D: `(L, S)` 或 3D: `(N*num_heads, L, S)` |
| 用途 | 标记 padding 位置 | 自定义注意力模式 |
| 值含义 | `True` = 忽略该位置 | `-inf` = 忽略该位置 |
| 性能 | 更高效 | 较慢（3D 时） |

其中：
- N = batch size
- L = target sequence length
- S = source sequence length

### 为什么之前没发现

在单元测试中使用的 batch size 较小（2），且 num_heads 也较小，所以维度问题不明显。
在实际训练中，batch_size=8, max_lanes可能很大（50-200），导致问题暴露。

## 日期

2025-11-20
