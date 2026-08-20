import math
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from timm.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

try:
    from .csms6s import CrossMerge, CrossScan, SelectiveScanCore
except:
    from csms6s import CrossMerge, CrossScan, SelectiveScanCore


class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(
            self.weight.shape
        )
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        x = x.permute(0, 3, 1, 2)
        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        channels_first=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


class mamba_init:
    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D


class DirectionalAudioVisualGate(nn.Module):
    def __init__(self, D, K=4, kernel_size=3, reduction=4):
        super().__init__()
        self.K = K
        self.D = D

        self.temporal_conv = nn.Conv1d(
            K * D, K * D, kernel_size, padding=kernel_size // 2, groups=K * D
        )
        self.audio_transform = nn.Conv1d(K * D, K * D // reduction, 1, groups=K)
        self.gate_fusion = nn.Sequential(
            nn.Conv1d(K * D // reduction, K * D // reduction, 1, groups=K),
            nn.ReLU(),
            nn.Conv1d(K * D // reduction, K * D, 1, groups=K),
        )

    def forward(self, audio):
        audio_temporal = self.temporal_conv(audio)
        audio_feat = self.audio_transform(audio_temporal)  # [B, K*D//r, L]
        gate = torch.sigmoid(self.gate_fusion(audio_feat))
        return gate, audio_temporal


class BiSTSSM_v2:
    def __initv2__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3,  # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()

        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        self.out_norm = LayerNorm(d_inner)

        self.forward_core = partial(
            self.forward_corev2,
            force_fp32=True,
            SelectiveScan=SelectiveScanCore,
            CrossScan=CrossScan,
            CrossMerge=CrossMerge,
        )
        k_group = 4

        d_proj = d_inner * 2
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.audio_proj_1 = Linear(256, 512)
        self.act: nn.Module = act_layer()

        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  # (K, N, inner)
        del self.x_proj

        self.out_act = nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.dt_projs = [
            self.dt_init(
                dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor
            )
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(
            d_state, d_inner, copies=k_group, merge=True
        )  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)

        self.audio_gate = DirectionalAudioVisualGate(512)

    def forward_corev2(
        self,
        x: torch.Tensor = None,
        audio_feat: torch.Tensor = None,
        # ==============================
        to_dtype=True,  # True: final out to dtype
        force_fp32=False,  # True: input fp32
        # ==============================
        ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=SelectiveScanCore,
        # SelectiveScan=SelectiveScanStateFn,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
        # no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        no_einsum=True,  # replace einsum with linear or conv1d to raise throughput
        # ==============================
        cascade2d=False,
        **kwargs,
    ):
        # print(SelectiveScan, CrossScan, CrossMerge, cascade2d)
        # <class 'lib.model.csms6s.SelectiveScanCore'> <class 'lib.model.csms6s.CrossScan'> <class 'lib.model.csms6s.CrossMerge'> False
        x_proj_weight = self.x_proj_weight  # [4, 36, 128]
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight  # [4, 128, 4]
        dt_projs_bias = self.dt_projs_bias  # [4, 128]
        A_logs = self.A_logs  # [512, 16]
        Ds = self.Ds  # [512]
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(
            u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True
        ):
            return SelectiveScan.apply(
                u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex
            )

        xs = CrossScan.apply(x)  # [8, 4, 128, 4131]
        if no_einsum:
            x_dbl = F.conv1d(
                xs.view(B, -1, L),
                x_proj_weight.view(-1, D, 1),
                bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None),
                groups=K,
            )  # [8, 144, 4131]
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            dts = F.conv1d(
                dts.contiguous().view(B, -1, L),
                dt_projs_weight.view(K * D, -1, 1),
                groups=K,
            )  # [8, 512, 4131]  B,C[8, 4, 16, 4131]
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L)  # [B, 512 (4*C), 4131]
        dts = dts.contiguous().view(B, -1, L)
        #! decomposed_audio T, fQ, LB, C

        #! fq, LB, C
        audio_feat = self.audio_proj_1(audio_feat)

        audio = audio_feat.new_empty((B, 4, D, L))
        audio[:, 0] = audio_feat.permute(2, 3, 0, 1).flatten(2, 3)
        audio[:, 1] = (
            audio_feat.permute(2, 3, 0, 1).transpose(dim0=2, dim1=3).flatten(2, 3)
        )
        audio[:, 2:4] = torch.flip(audio[:, 0:2], dims=[-1])
        audio = audio.view(B, K * D, -1)

        joint_gate, audio_temporal = self.audio_gate(audio)

        #! C, fq, LB
        dts = (dts + joint_gate * audio_temporal).contiguous()
        # dts = (dts + audio).contiguous()
        As = -torch.exp(A_logs.to(torch.float))
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)  # [B, 4, 16, H*W]
        Ds = Ds.to(torch.float)  # (K * c)  #[512]
        delta_bias = dt_projs_bias.view(-1).to(torch.float)  # [512]

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)  # [8, 4, 128, H, W]
        y: torch.Tensor = CrossMerge.apply(ys)  # [8, 128, 4131]

        if getattr(self, "__DEBUG__", False):
            setattr(
                self,
                "__data__",
                dict(
                    A_logs=A_logs,
                    Bs=Bs,
                    Cs=Cs,
                    Ds=Ds,
                    us=xs,
                    dts=dts,
                    delta_bias=delta_bias,
                    ys=ys,
                    y=y,
                ),
            )

        y = y.view(B, -1, H, W)

        y = out_norm(y.permute(0, 2, 3, 1)).permute(1, 2, 0, 3)

        return y.to(x.dtype) if to_dtype else y

    "Here is the function of s6"

    def forwardv2(self, x, audio_feat):
        # x = x.permute(2, 3, 0, 1).contiguous()
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        z = self.act(z)
        if not self.channel_first:
            x = x.permute(2, 3, 0, 1).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)
        x = self.act(x)  # [B, C, 243, 17]

        y = self.forward_core(x, audio_feat)

        y = self.out_act(y)
        y = y * z
        out = self.dropout(self.out_proj(y))

        return out

    def A_log_init(self, d_state, d_inner, copies, merge):
        pass


class BiSTSSM(nn.Module, mamba_init, BiSTSSM_v2):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3,  # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            act_layer=act_layer,
            d_conv=d_conv,
            conv_bias=conv_bias,
            dropout=dropout,
            bias=bias,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            initialize=initialize,
            forward_type=forward_type,
            channel_first=channel_first,
        )
        self.__initv2__(**kwargs)


class BiSTSSMBlock(nn.Module):
    """
    A Vision Mamba Block that supports both Pre-Normalization and Post-Normalization.

    Args:
        pre_norm (bool): If True, applies normalization before the main operation (Pre-Norm).
                         If False, applies normalization after the residual connection (Post-Norm).
                         Defaults to True.
        ... (other arguments)
    """

    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0.0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio: float = 4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        normalize_before: bool = True,  # Use this to control normalization style
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.normalize_before = normalize_before

        if self.ssm_branch:
            self.norm1 = norm_layer(hidden_dim)
            self.mamba = BiSTSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            _MLP = Mlp
            self.mlp = _MLP(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                channels_first=channel_first,
            )

    def _forward_pre_norm(self, frame_query: torch.Tensor, audio_feat: torch.Tensor):
        # Pre-Normalization: x -> Norm -> Mamba/MLP -> DropPath -> + -> ...
        if self.ssm_branch:
            frame_query = frame_query + self.drop_path(
                self.mamba(self.norm1(frame_query), audio_feat)
            )
        if self.mlp_branch:
            frame_query = frame_query + self.drop_path(
                self.mlp(self.norm2(frame_query))
            )
        return frame_query

    def _forward_post_norm(self, frame_query: torch.Tensor, audio_feat: torch.Tensor):
        # Post-Normalization: x -> Mamba/MLP -> DropPath -> + -> Norm -> ...
        if self.ssm_branch:
            frame_query = self.norm1(
                frame_query + self.drop_path(self.mamba(frame_query, audio_feat))
            )
        if self.mlp_branch:
            frame_query = self.norm2(
                frame_query + self.drop_path(self.mlp(frame_query))
            )
        return frame_query

    def forward(self, frame_query: torch.Tensor, audio_feat: torch.Tensor):
        """
        Forward pass for the BiSTSSMBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        if self.normalize_before:
            return self._forward_pre_norm(frame_query, audio_feat)
        else:
            return self._forward_post_norm(frame_query, audio_feat)
