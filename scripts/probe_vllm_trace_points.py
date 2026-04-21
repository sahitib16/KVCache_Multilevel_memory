#!/usr/bin/env python3
"""Probe the installed vLLM package for trace-capture hook points.

This script does not require or start a vLLM engine. It only checks whether the
current environment exposes the modules and methods we would use for block-level
shadow logging.
"""

from __future__ import annotations

import importlib
import inspect


def probe_module(name: str):
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"{name}: unavailable ({type(exc).__name__}: {exc})")
        return None
    print(f"{name}: available")
    return module


def print_attrs(obj, attrs: list[str]) -> None:
    for attr in attrs:
        value = getattr(obj, attr, None)
        status = "yes" if value is not None else "no"
        print(f"  {attr}: {status}")
        if value is not None and callable(value):
            try:
                signature = inspect.signature(value)
            except (TypeError, ValueError):
                signature = "(signature unavailable)"
            print(f"    signature: {signature}")


def main() -> None:
    vllm = probe_module("vllm")
    if vllm is not None:
        print(f"vllm version: {getattr(vllm, '__version__', 'unknown')}")

    kv_module = probe_module("vllm.v1.core.kv_cache_manager")
    if kv_module is not None:
        manager = getattr(kv_module, "KVCacheManager", None)
        blocks = getattr(kv_module, "KVCacheBlocks", None)
        print("KVCacheManager methods:")
        if manager is not None:
            print_attrs(
                manager,
                [
                    "get_block_ids",
                    "get_blocks",
                    "get_computed_blocks",
                    "free",
                    "evict_blocks",
                    "cache_blocks",
                    "allocate_slots",
                ],
            )
        else:
            print("  KVCacheManager: missing")
        print("KVCacheBlocks methods:")
        if blocks is not None:
            print_attrs(blocks, ["get_block_ids", "get_unhashed_block_ids", "get_unhashed_block_ids_all_groups"])
        else:
            print("  KVCacheBlocks: missing")

    block_table_module = probe_module("vllm.v1.worker.block_table")
    if block_table_module is not None:
        block_table = getattr(block_table_module, "BlockTable", None)
        multi_group = getattr(block_table_module, "MultiGroupBlockTable", None)
        print("BlockTable methods:")
        if block_table is not None:
            print_attrs(
                block_table,
                [
                    "append_row",
                    "add_row",
                    "clear_row",
                    "move_row",
                    "swap_row",
                    "compute_slot_mapping",
                    "commit_block_table",
                    "get_cpu_tensor",
                    "get_device_tensor",
                    "get_numpy_array",
                    "map_to_kernel_blocks",
                ],
            )
        else:
            print("  BlockTable: missing")
        print("MultiGroupBlockTable methods:")
        if multi_group is not None:
            print_attrs(
                multi_group,
                [
                    "append_row",
                    "add_row",
                    "clear_row",
                    "move_row",
                    "swap_row",
                    "compute_slot_mapping",
                    "commit_block_table",
                    "clear",
                ],
            )
        else:
            print("  MultiGroupBlockTable: missing")


if __name__ == "__main__":
    main()
