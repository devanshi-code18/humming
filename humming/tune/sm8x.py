import math

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.device import current_device
from humming.tune.base import DeviceHeuristics
from humming.utils.math import ceil_div
from humming.utils.smem import estimate_smem_size_layer


class Sm80Heuristics(DeviceHeuristics):
    max_smem_size: int = 163 * 1024
    sm_version: int = 80
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16, dtypes.bfloat16]
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8]
    b4_allowed_dtypes: list[dtypes.DataType] = [dtypes.int4]

    @classmethod
    def get_base_config(
        cls,
        a_dtype: dtypes.DataType,
        b_dtype: dtypes.DataType,
        group_size: int,
        use_f16_accum: bool,
        use_fused_e8m0_scale: bool,
        gemm_type: GemmType,
        shape_k: int,
    ):
        has_raw_weight = a_dtype.num_bits == b_dtype.num_bits
        if a_dtype.num_bits == 16 or group_size == 0 and has_raw_weight:
            if use_f16_accum:
                return {
                    "block_shape": (128, 256, 1024 // a_dtype.num_bits),
                    "warp_shape": (128, 64, 512 // a_dtype.num_bits),
                }
            else:
                return {
                    "block_shape": (128, 256, 1024 // a_dtype.num_bits),
                    "warp_shape": (64, 64, 1024 // a_dtype.num_bits),
                }
        elif group_size == 0:
            return {
                "block_shape": (128, 256, 1024 // a_dtype.num_bits),
                "warp_shape": (128, 32, 1024 // a_dtype.num_bits),
            }
        elif group_size > 0:
            return {
                "block_shape": (64, 128, 1024 // a_dtype.num_bits),
                "warp_shape": (64, 32, 1024 // a_dtype.num_bits),
                "num_ctas_per_sm": 2,
            }

    @classmethod
    def get_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
        use_m_major_input_scale: bool = False,
    ):
        config = super().get_config(
            layer_config,
            shape_m,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        if cls._should_use_decode_tile(layer_config, shape_m, use_f16_accum, use_batch_invariant, gemm_type):
            return cls._get_decode_tile_config(config, layer_config, shape_m)
        if cls._should_use_mid_m_tile(layer_config, shape_m, use_f16_accum, use_batch_invariant, gemm_type):
            return cls._get_mid_m_tile_config(config, layer_config, shape_m)
        return config

    @classmethod
    def _should_use_decode_tile(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool,
        use_batch_invariant: bool,
        gemm_type: GemmType,
    ) -> bool:
        group_size = layer_config.input_scale_group_size or layer_config.weight_scale_group_size
        shape_k = layer_config.shape_k
        is_dense_w4a16 = gemm_type == GemmType.DENSE and layer_config.a_dtype.num_bits == 16
        is_dense_w4a16 &= layer_config.b_dtype.num_bits == 4 and group_size == 128
        is_decode_size = (shape_m <= 16 and shape_k <= 5120) or (shape_m <= 32 and shape_k <= 4096)
        is_tile_aligned = layer_config.shape_n % 64 == 0 and shape_k % 128 == 0
        uses_default_math = not use_f16_accum and not use_batch_invariant
        is_sm80 = cls.sm_version == 80
        return is_sm80 and is_dense_w4a16 and is_decode_size and is_tile_aligned and uses_default_math

    @classmethod
    def _get_decode_tile_config(cls, config: dict, layer_config: LayerConfig, shape_m: int) -> dict:
        """Give each CTA one whole output tile for decode-size W4A16 GEMMs.

        Stream-K splits a tile's K range across CTAs, and every split adds a
        pipeline refill and a global-memory reduction: about 5 us on A100,
        while the weights of a 4096x4096 layer stream in about 6 us. A
        data-parallel grid of one wave is faster even when some SMs idle.
        Layers with K > 5120 keep Stream-K: their tiles are long enough to
        hide the reduction, and the data-parallel grid lost there.
        """
        num_sms = current_device.sm_count
        num_ctas_per_sm = 2
        max_num_tiles = num_sms * num_ctas_per_sm
        # At M <= 8 a tile is cheap, so fill every SM; at larger M a 128-column
        # tile pays off once it covers 55% of the SMs (56 tiles lost at N=7168).
        min_num_tiles = num_sms if shape_m <= 8 else math.ceil(0.55 * num_sms)

        shape_n = layer_config.shape_n
        one_wave_block_n = [
            block_n for block_n in (128, 64) if shape_n % block_n == 0 and shape_n // block_n <= max_num_tiles
        ]
        filling_block_n = [block_n for block_n in one_wave_block_n if shape_n // block_n >= min_num_tiles]
        if filling_block_n:
            block_n = filling_block_n[0]
        elif one_wave_block_n and shape_m <= 16:
            # Up to M = 16 a 64-column grid wins even when it fills under half the SMs.
            block_n = one_wave_block_n[-1]
        else:
            return config

        block_m = 8 if shape_m <= 8 else 16 if shape_m <= 16 else 32
        warp_n = 64
        block_k = 128
        warp_k = block_k * (block_n // warp_n) // 4
        block_shape = (block_m, block_n, block_k)
        warp_shape = (block_m, warp_n, warp_k)
        num_stages = 6 if block_n == 64 else 4

        smem_size = estimate_smem_size_layer(
            layer_config,
            block_shape,
            GemmType.DENSE,
            num_stages,
            warp_shape=warp_shape,
            output_chunk_rows=config.get("output_chunk_rows", 0),
        )
        if smem_size * num_ctas_per_sm > cls.max_smem_size:
            return config

        config = dict(config)
        config.update(
            block_shape=block_shape,
            warp_shape=warp_shape,
            use_stream_k=False,
            num_stages=num_stages,
            num_ctas_per_sm=num_ctas_per_sm,
            num_sms=ceil_div(shape_n // block_n, num_ctas_per_sm),
        )
        return config

    # Measured best (block_k, warp_n, warp_k, num_stages) per data-parallel (block_m, block_n) tile.
    _MID_M_TILE_SCHEDULES = {
        (64, 64): (64, 32, 32, 5),
        (96, 64): (64, 32, 32, 5),
        (128, 64): (64, 64, 32, 5),
        (32, 128): (128, 64, 32, 3),
        (64, 128): (128, 64, 32, 3),
        (96, 128): (128, 32, 64, 3),
        (128, 128): (64, 64, 64, 3),
        (32, 256): (128, 64, 64, 3),
        (64, 256): (64, 64, 32, 3),
        (96, 256): (64, 32, 64, 3),
        (128, 256): (64, 64, 64, 5),
    }

    @classmethod
    def _should_use_mid_m_tile(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool,
        use_batch_invariant: bool,
        gemm_type: GemmType,
    ) -> bool:
        group_size = layer_config.input_scale_group_size or layer_config.weight_scale_group_size
        is_dense_w4a16 = gemm_type == GemmType.DENSE and layer_config.a_dtype.num_bits == 16
        is_dense_w4a16 &= layer_config.b_dtype.num_bits == 4 and group_size == 128
        is_mid_m = 32 < shape_m <= 128 and layer_config.shape_k <= 5120
        is_tile_aligned = layer_config.shape_n % 64 == 0 and layer_config.shape_k % 128 == 0
        uses_default_math = not use_f16_accum and not use_batch_invariant
        is_sm80 = cls.sm_version == 80
        return is_sm80 and is_dense_w4a16 and is_mid_m and is_tile_aligned and uses_default_math

    @classmethod
    def _get_mid_m_tile_config(cls, config: dict, layer_config: LayerConfig, shape_m: int) -> dict:
        """Use one wave of whole tiles at 32 < M <= 128 when it fills the GPU.

        Same trade-off as the decode tiles: Stream-K reductions cost more than
        they save once whole tiles cover at least 85% of the SMs in one wave.
        Fewer M blocks first (each re-reads the weights), then the larger tile,
        then less M padding.
        """
        num_sms = current_device.sm_count
        shape_n = layer_config.shape_n
        candidates = []
        for (block_m, block_n), schedule in cls._MID_M_TILE_SCHEDULES.items():
            num_m_blocks = ceil_div(shape_m, block_m)
            num_tiles = num_m_blocks * (shape_n // block_n)
            padding = num_m_blocks * block_m / shape_m - 1
            is_one_full_wave = 0.85 * num_sms <= num_tiles <= num_sms
            if shape_n % block_n == 0 and is_one_full_wave and padding <= 0.34:
                tile_work = block_n * shape_m / num_m_blocks
                candidates.append((-num_m_blocks, tile_work, -padding, block_m, block_n, schedule, num_tiles))
        if not candidates:
            return config
        _, _, _, block_m, block_n, (block_k, warp_n, warp_k, num_stages), num_tiles = max(candidates)

        warp_m = block_m if block_m <= 96 else 64
        block_shape = (block_m, block_n, block_k)
        warp_shape = (warp_m, warp_n, warp_k)
        smem_size = estimate_smem_size_layer(
            layer_config,
            block_shape,
            GemmType.DENSE,
            num_stages,
            warp_shape=warp_shape,
            output_chunk_rows=config.get("output_chunk_rows", 0),
        )
        if smem_size > cls.max_smem_size:
            return config

        config = dict(config)
        config.update(
            block_shape=block_shape,
            warp_shape=warp_shape,
            use_stream_k=False,
            num_stages=num_stages,
            num_ctas_per_sm=1,
            num_sms=num_tiles,
        )
        return config


class Sm86Heuristics(DeviceHeuristics):
    max_smem_size: int = 99 * 1024
    sm_version: int = 86
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16, dtypes.bfloat16]
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8]
    b4_allowed_dtypes: list[dtypes.DataType] = [dtypes.int4]

    @classmethod
    def get_base_config(
        cls,
        a_dtype: dtypes.DataType,
        b_dtype: dtypes.DataType,
        group_size: int,
        use_f16_accum: bool,
        use_fused_e8m0_scale: bool,
        gemm_type: GemmType,
        shape_k: int,
    ):
        if a_dtype.num_bits == 16 and use_f16_accum:
            num_stages = 3 if b_dtype.num_bits < 8 else 2
            return {
                "block_shape": (128, 256, 64),
                "warp_shape": (64, 64, 64),
                "num_stages": num_stages,
            }
        elif a_dtype.num_bits == 16 and not use_f16_accum:
            num_stages = 3 if b_dtype.num_bits < 8 else 2
            return {
                "block_shape": (128, 256, 64),
                "warp_shape": (64, 64, 64),
                "num_stages": num_stages,
            }
        elif use_fused_e8m0_scale:
            return {
                "block_shape": (128, 128, 512 // a_dtype.num_bits),
                "warp_shape": (128, 32, 512 // a_dtype.num_bits),
                "num_stages": 2,
                "num_warps_per_sm": 2,
            }
        elif group_size == 0 and a_dtype.num_bits == b_dtype.num_bits:
            return {
                "block_shape": (128, 256, 512 // a_dtype.num_bits),
                "warp_shape": (64, 64, 512 // a_dtype.num_bits),
                "num_stages": 2,
            }
        elif group_size == 0:
            return {
                "block_shape": (128, 256, 1024 // a_dtype.num_bits),
                "warp_shape": (128, 32, 1024 // a_dtype.num_bits),
                "num_stages": 2,
            }
        else:
            return {
                "block_shape": (64, 256, 1024 // a_dtype.num_bits),
                "warp_shape": (64, 32, 1024 // a_dtype.num_bits),
                "num_stages": 2,
            }


class Sm87Heuristics(Sm80Heuristics):
    sm_version: int = 87


class Sm89Heuristics(Sm86Heuristics):
    sm_version: int = 89
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8, dtypes.float8e4m3, dtypes.float8e5m2]
