import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable


# SMM_QmK 和 SMM_AmV 类保持不变
class SMM_QmK(Function):
    @staticmethod
    def forward(ctx, A, B, index):
        ctx.save_for_backward(A, B, index)
        import smm_cuda

        A_float = A.float() if A.dtype == torch.float16 else A
        B_float = B.float() if B.dtype == torch.float16 else B

        result = smm_cuda.SMM_QmK_forward_cuda(
            A_float.contiguous(),
            B_float.contiguous(),
            index.contiguous()
        )

        if A.dtype == torch.float16:
            result = result.half()

        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        A, B, index = ctx.saved_tensors
        import smm_cuda

        grad_float = grad_output.float() if grad_output.dtype == torch.float16 else grad_output
        A_float = A.float() if A.dtype == torch.float16 else A
        B_float = B.float() if B.dtype == torch.float16 else B

        grad_A, grad_B = smm_cuda.SMM_QmK_backward_cuda(
            grad_float.contiguous(),
            A_float.contiguous(),
            B_float.contiguous(),
            index.contiguous()
        )

        if A.dtype == torch.float16:
            grad_A = grad_A.half()
        if B.dtype == torch.float16:
            grad_B = grad_B.half()

        return grad_A, grad_B, None


class SMM_AmV(Function):
    @staticmethod
    def forward(ctx, A, B, index):
        ctx.save_for_backward(A, B, index)
        import smm_cuda

        A_float = A.float() if A.dtype == torch.float16 else A
        B_float = B.float() if B.dtype == torch.float16 else B

        result = smm_cuda.SMM_AmV_forward_cuda(
            A_float.contiguous(),
            B_float.contiguous(),
            index.contiguous()
        )

        if A.dtype == torch.float16:
            result = result.half()

        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        A, B, index = ctx.saved_tensors
        import smm_cuda

        grad_float = grad_output.float() if grad_output.dtype == torch.float16 else grad_output
        A_float = A.float() if A.dtype == torch.float16 else A
        B_float = B.float() if B.dtype == torch.float16 else B

        grad_A, grad_B = smm_cuda.SMM_AmV_backward_cuda(
            grad_float.contiguous(),
            A_float.contiguous(),
            B_float.contiguous(),
            index.contiguous()
        )

        if A.dtype == torch.float16:
            grad_A = grad_A.half()
        if B.dtype == torch.float16:
            grad_B = grad_B.half()

        return grad_A, grad_B, None


class SemanticAwareSparseAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_clusters=8, dropout=0.0,
                 kmeans_max_iter=10, top_p=0.8, use_sparse_kernel=True):  # 减少了默认的kmeans_max_iter
        """
        语义感知的稀疏交叉注意力模块 - 并行多头版本
        """
        super(SemanticAwareSparseAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.head_dim = embed_dim // num_heads
        self.kmeans_max_iter = kmeans_max_iter
        self.top_p = top_p
        self.use_sparse_kernel = use_sparse_kernel

        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def kmeans_plusplus_init(self, features, n_clusters):
        """
        K-means++初始化聚类中心(GPU版本) - 批量并行.
        这个函数仍然有循环，但由于 n_clusters 较小，其开销远小于其他瓶颈。
        """
        batch_size, seq_len, head_dim = features.shape
        device = features.device

        n_clusters = min(n_clusters, seq_len)

        centers = torch.zeros(batch_size, n_clusters, head_dim, device=device, dtype=features.dtype)

        # 批量随机选择第一个中心
        first_indices = torch.randint(0, seq_len, (batch_size,), device=device)
        batch_indices = torch.arange(batch_size, device=device)
        centers[:, 0] = features[batch_indices, first_indices]

        # [B, seq_len]
        min_distances = torch.cdist(features, centers[:, 0:1]).squeeze(2)

        for i in range(1, n_clusters):
            # 距离的平方作为概率分布
            probs = min_distances ** 2

            # 批量采样下一个中心
            next_indices = torch.multinomial(probs, 1).squeeze(1)  # [B]
            centers[:, i] = features[batch_indices, next_indices]

            # 更新到最近中心的距离
            new_distances = torch.cdist(features, centers[:, i:i + 1]).squeeze(2)
            min_distances = torch.minimum(min_distances, new_distances)

        return centers

    def build_block_sparse_indices_vectorized(self, q_labels_sorted, k_labels_sorted, selected_mask):
        """
        【块稀疏 & 完全向量化版 - 已修正】构建稀疏注意力索引

        Args:
            q_labels_sorted: [batch_size, N_q] (已按簇排序)
            k_labels_sorted: [batch_size, N_k] (已按簇排序)
            selected_mask: [batch_size, n_q_clusters, n_k_clusters] (bool)

        Returns:
            sparse_indices: [batch_size, N_q, topk]
            actual_topk: int
        """
        batch_size, q_len = q_labels_sorted.shape
        k_len = k_labels_sorted.shape[1]
        n_q_clusters, n_k_clusters = selected_mask.shape[1], selected_mask.shape[2]
        device = q_labels_sorted.device

        # === 1. 向量化计算 K 簇的起始位置和大小 ===

        # a. 找到所有 K 簇的边界 (不使用循环)
        k_padding = torch.full((batch_size, 1), -1, device=device, dtype=k_labels_sorted.dtype)
        k_labels_padded = torch.cat([k_padding, k_labels_sorted], dim=1)
        k_is_start = (k_labels_padded[:, 1:] != k_labels_padded[:, :-1])

        # b. 创建 K 簇元数据查找表 (Lookup Table)
        k_start_map = torch.full((batch_size, n_k_clusters), -1, dtype=torch.long, device=device)
        k_size_map = torch.zeros((batch_size, n_k_clusters), dtype=torch.long, device=device)

        k_start_coords = k_is_start.nonzero(as_tuple=False)
        if k_start_coords.numel() > 0:  # 确保至少有一个簇
            k_start_labels = k_labels_sorted[k_start_coords[:, 0], k_start_coords[:, 1]]
            k_start_map[k_start_coords[:, 0], k_start_labels] = k_start_coords[:, 1]

        batch_offset = torch.arange(batch_size, device=device) * n_k_clusters
        unique_b_kc_ids = batch_offset.unsqueeze(1) + k_labels_sorted
        counts = torch.bincount(unique_b_kc_ids.flatten(), minlength=batch_size * n_k_clusters)
        k_size_map = counts.view(batch_size, n_k_clusters)

        # === 2. 将 K 簇元数据广播给每个 Query 并根据 selected_mask 筛选 ===

        expanded_k_starts = k_start_map.unsqueeze(1).expand(-1, q_len, -1)
        expanded_k_sizes = k_size_map.unsqueeze(1).expand(-1, q_len, -1)

        q_allowed_k_mask = selected_mask.gather(1, q_labels_sorted.unsqueeze(-1).expand(-1, -1, n_k_clusters))

        query_k_starts = expanded_k_starts.where(q_allowed_k_mask, -1)
        query_k_sizes = expanded_k_sizes.where(q_allowed_k_mask, 0)

        # === 3. 并行生成索引并使用 topk 技巧压缩 ===

        actual_topk = query_k_sizes.sum(dim=-1).max().item()
        actual_topk = max(1, actual_topk)

        max_k_block_size = k_size_map.max().item() if k_size_map.numel() > 0 else 0
        max_k_block_size = max(1, max_k_block_size)  # 确保至少为1

        rel_indices = torch.arange(max_k_block_size, device=device)

        candidate_indices = query_k_starts.unsqueeze(-1) + rel_indices
        valid_mask = rel_indices < query_k_sizes.unsqueeze(-1)
        masked_indices = candidate_indices.where(valid_mask, k_len)
        flat_masked_indices = masked_indices.flatten(start_dim=2)
        sparse_indices = torch.topk(flat_masked_indices, k=actual_topk, dim=-1, largest=False).values

        # === 4. 清理哨兵值 ===
        sparse_indices[sparse_indices == k_len] = 0

        return sparse_indices.int(), actual_topk

    def kmeans_clustering_gpu_optimized(self, features, n_clusters):
        """
        【优化版】GPU K-means聚类 - 完全向量化

        Args:
            features: [batch_size, seq_len, head_dim]

        Returns:
            labels: [batch_size, seq_len]
            indices: [batch_size, seq_len] (排序后的索引)
            centers: [batch_size, n_clusters, head_dim]
        """
        batch_size, seq_len, head_dim = features.shape
        n_clusters = min(n_clusters, seq_len)

        centers = self.kmeans_plusplus_init(features, n_clusters)

        for _ in range(self.kmeans_max_iter):
            distances = torch.cdist(features, centers)
            labels = distances.argmin(dim=2)
            one_hot_labels = F.one_hot(labels, num_classes=n_clusters).to(features.dtype)  # [B, seq_len, n_clusters]
            cluster_sizes = one_hot_labels.sum(dim=1).clamp(min=1)
            cluster_sums = torch.einsum('bnc,bnd->bcd', one_hot_labels, features)
            new_centers = cluster_sums / cluster_sizes.unsqueeze(-1)
            centers = new_centers

        distances = torch.cdist(features, centers)
        labels = distances.argmin(dim=2)
        indices = torch.argsort(labels, dim=1)

        return labels, indices, centers

    def compute_cluster_importance(self, q_centers, k_centers):
        similarity_scores = torch.bmm(q_centers, k_centers.transpose(-2, -1)) * self.scale
        return similarity_scores

    def compute_cluster_sizes(self, labels, n_clusters):
        one_hot = F.one_hot(labels, num_classes=n_clusters)
        return one_hot.sum(dim=1)

    def top_p_cluster_selection(self, importance_scores, k_cluster_sizes, top_p=0.9):
        exp_scores = torch.exp(importance_scores)
        k_sizes_expanded = k_cluster_sizes.unsqueeze(1)
        weighted_scores = k_sizes_expanded * exp_scores
        cluster_probs = weighted_scores / (weighted_scores.sum(dim=2, keepdim=True) + 1e-8)

        sorted_probs, sorted_indices = torch.sort(cluster_probs, dim=2, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=2)

        # 找到超过top_p的位置，并确保至少选择一个
        selected_mask_sorted = cumsum_probs <= top_p
        selected_mask_sorted[:, :, 0] = True

        # scatter回原始索引位置
        selected_mask = torch.zeros_like(cluster_probs, dtype=torch.bool)
        selected_mask.scatter_(2, sorted_indices, selected_mask_sorted)

        return selected_mask, cluster_probs

    def reorder_tokens(self, tokens, indices):
        return torch.gather(tokens, 1, indices.unsqueeze(2).expand_as(tokens))

    def restore_order(self, tokens, indices):
        inverse_indices = torch.empty_like(indices)
        inverse_indices.scatter_(1, indices,
                                 torch.arange(indices.shape[1], device=indices.device).unsqueeze(0).expand_as(indices))
        return torch.gather(tokens, 1, inverse_indices.unsqueeze(2).expand_as(tokens))

    def sparse_softmax(self, attn_scores):
        return F.softmax(attn_scores, dim=-1)

    def forward(self, query, key, value, query_pos=None, return_attention_info=False, spatial_shapes=None):
        q_len, num_frame, embed_dim = query.shape
        k_len = key.shape[0]

        if query_pos is not None:
            query = query + query_pos

        Q_proj = self.q_proj(query)
        K_proj = self.k_proj(key)
        V_proj = self.v_proj(value)

        Q = Q_proj.view(q_len, num_frame, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        K = K_proj.view(k_len, num_frame, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        V = V_proj.view(k_len, num_frame, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        batch_size = num_frame * self.num_heads
        Q = Q.reshape(batch_size, q_len, self.head_dim)
        K = K.reshape(batch_size, k_len, self.head_dim)
        V = V.reshape(batch_size, k_len, self.head_dim)

        q_labels, q_indices, q_centers = self.kmeans_clustering_gpu_optimized(Q, self.num_clusters)
        k_labels, k_indices, k_centers = self.kmeans_clustering_gpu_optimized(K, self.num_clusters)

        similarity_scores = self.compute_cluster_importance(q_centers, k_centers)

        k_cluster_sizes = self.compute_cluster_sizes(k_labels, self.num_clusters)

        selected_mask, cluster_probs = self.top_p_cluster_selection(
            similarity_scores, k_cluster_sizes, self.top_p
        )

        # 8. 并行重排序
        Q_reordered = self.reorder_tokens(Q, q_indices)
        K_reordered = self.reorder_tokens(K, k_indices)
        V_reordered = self.reorder_tokens(V, k_indices)

        # 9. 获取排序后的标签
        q_labels_sorted = torch.gather(q_labels, 1, q_indices)
        k_labels_sorted = torch.gather(k_labels, 1, k_indices)

        # 10. 并行构建稀疏索引
        sparse_indices, actual_topk = self.build_block_sparse_indices_vectorized(
            q_labels_sorted, k_labels_sorted, selected_mask
        )

        # 11. 并行稀疏注意力计算
        attn_output = None
        if self.use_sparse_kernel:
            try:
                K_transposed = K_reordered.transpose(-2, -1)
                attn_scores = SMM_QmK.apply(Q_reordered, K_transposed, sparse_indices) * self.scale
                attn_weights = self.sparse_softmax(attn_scores)
                attn_weights = self.dropout(attn_weights)
                attn_output = SMM_AmV.apply(attn_weights, V_reordered, sparse_indices)
            except Exception as e:
                print(f"Warning: Sparse kernel failed with error: {e}. Falling back to dense attention.")
                self.use_sparse_kernel = False  # Avoid trying again

        if attn_output is None:  # Fallback logic
            # 使用稀疏索引进行 gather 操作
            batch_size_fallback, q_seq_len, _ = Q_reordered.shape

            K_gathered = K_reordered.gather(1, sparse_indices.view(batch_size_fallback, -1).unsqueeze(-1).expand(-1, -1,
                                                                                                                 self.head_dim)).view(
                batch_size_fallback, q_seq_len, actual_topk, self.head_dim)
            V_gathered = V_reordered.gather(1, sparse_indices.view(batch_size_fallback, -1).unsqueeze(-1).expand(-1, -1,
                                                                                                                 self.head_dim)).view(
                batch_size_fallback, q_seq_len, actual_topk, self.head_dim)

            attn_scores = torch.matmul(Q_reordered.unsqueeze(2), K_gathered.transpose(-2, -1)).squeeze(2) * self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            attn_output = torch.matmul(attn_weights.unsqueeze(2), V_gathered).squeeze(2)

        # Get spatial dimensions for visualization
        # Use provided spatial_shapes if available, otherwise infer from k_len
        if spatial_shapes is not None:
            H_spatial, W_spatial = spatial_shapes
        else:
            # Fallback: infer from k_len (may cause visualization artifacts)
            N_k_total = k_len
            H_spatial = W_spatial = int(N_k_total ** 0.5)
            if H_spatial * W_spatial != N_k_total:
                for i in range(int(N_k_total**0.5), 0, -1):
                    if N_k_total % i == 0:
                        H_spatial, W_spatial = i, N_k_total // i
                        break
        
        # 12. 并行恢复原始顺序
        attn_output = self.restore_order(attn_output, q_indices)  # shape: (num_frame * num_heads, q_len, head_dim)

        # [MODIFIED] 核心修改：将输出张量恢复为原始形状
        # 目标: (num_frame * num_heads, q_len, head_dim) -> (q_len, num_frame, embed_dim)

        # 1. 分离批次维度
        # shape: (num_frame, num_heads, q_len, head_dim)
        attn_output = attn_output.view(num_frame, self.num_heads, q_len, self.head_dim)

        # 2. 置换维度，将 q_len 和 num_frame 移到前面
        # shape: (q_len, num_frame, num_heads, head_dim)
        attn_output = attn_output.permute(2, 0, 1, 3)

        # 3. 合并 head 维度
        # shape: (q_len, num_frame, embed_dim)
        attn_output = attn_output.contiguous().view(q_len, num_frame, embed_dim)

        # 4. 最终投影
        output = self.out_proj(attn_output)

        if return_attention_info:
            # 计算用于可视化的聚类相似度热力图
            with torch.no_grad():
                # 计算完整的聚类间相似度矩阵 [num_frame, num_heads, n_clusters, n_clusters]
                # q_centers: [batch_size, n_clusters, head_dim] where batch_size = num_frame * num_heads
                # 需要 reshape 为 [num_frame, num_heads, n_clusters, head_dim]
                # Note: num_frame here is the actual number of video frames in the batch
                num_frame_actual = q_centers.shape[0] // self.num_heads
                q_centers_reshaped = q_centers.view(num_frame_actual, self.num_heads, self.num_clusters, self.head_dim)
                k_centers_reshaped = k_centers.view(num_frame_actual, self.num_heads, self.num_clusters, self.head_dim)
                
                # 计算相似度: [num_frame, num_heads, n_clusters, n_clusters]
                cluster_similarity = torch.einsum('fhnc,fhmc->fhnm', q_centers_reshaped, k_centers_reshaped) * self.scale
            
            # Use the spatial dimensions inferred from k_len (H_spatial, W_spatial)
            
            return output, {
                'sparse_indices': sparse_indices,
                'selected_mask': selected_mask,
                'cluster_probs': cluster_probs,
                'topk': actual_topk,
                'q_centers': q_centers,
                'k_centers': k_centers,
                'q_labels': q_labels,
                'k_labels': k_labels,
                'q_indices': q_indices,
                'k_indices': k_indices,  # Use k_indices to map sorted labels back to original positions
                'cluster_similarity': cluster_similarity,
                'similarity_scores': similarity_scores,
                # [MODIFIED] Sparsity is calculated per-item, based on k_len
                'sparsity': 1 - actual_topk / k_len if k_len > 0 else 0,
                'num_heads': self.num_heads,
                'num_clusters': self.num_clusters,
                'head_dim': self.head_dim,
                'spatial_size': (H_spatial, W_spatial),
            }

        return output


if __name__ == "__main__":
    num_frame = 5
    C = 256
    seq_len = 300

    attention = SemanticAwareSparseAttention(
        embed_dim=C,
        num_heads=8,
        num_clusters=8,
        dropout=0.1,
        kmeans_max_iter=50,
        top_p=0.8,
        use_sparse_kernel=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attention = attention.to(device)

    query = torch.randn(100, num_frame, C).to(device)
    key = torch.randn(seq_len, num_frame, C).to(device)
    value = torch.randn(seq_len, num_frame, C).to(device)

    output, all_head_info = attention(query, key, value, return_attention_info=True)
    print(output.shape)