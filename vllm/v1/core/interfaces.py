# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Callable, Optional

from vllm.distributed.kv_events import KVCacheEvent
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request


class AbstractBlockPool(ABC):

    @abstractmethod
    def get_cached_block(self,
                         block_hash: BlockHashType) -> Optional[KVCacheBlock]:
        pass

    @abstractmethod
    def cache_full_blocks(
            self,
            request: Request,
            blocks: list[KVCacheBlock],
            block_hashes: list[BlockHashType],
            num_cached_blocks: int,
            num_full_blocks: int,
            block_size: int,
            hash_fn: Callable,
    ) -> None:
        pass

    @abstractmethod
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        pass

    @abstractmethod
    def touch(self, blocks: list[KVCacheBlock]) -> None:
        pass

    @abstractmethod
    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        pass

    @abstractmethod
    def reset_prefix_cache(self) -> bool:
        pass

    @abstractmethod
    def get_num_free_blocks(self) -> int:
        pass

    @abstractmethod
    def get_usage(self) -> float:
        pass

    @abstractmethod
    def take_events(self) -> list[KVCacheEvent]:
        pass

class AbstractKVCacheManager(ABC):

    @property
    @abstractmethod
    def usage(self) -> float:
        pass

    @abstractmethod
    def make_prefix_cache_stats(self) -> Optional[PrefixCacheStats]:
        pass

    @abstractmethod
    def get_computed_blocks(
            self, request: Request) -> tuple[list[KVCacheBlock], int]:
        pass

    @abstractmethod
    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
        num_lookahead_tokens: int = 0,
    ) -> Optional[list[KVCacheBlock]]:
        pass

    @abstractmethod
    def free(self, request: Request) -> None:
        pass

    @abstractmethod
    def reset_prefix_cache(self) -> bool:
        pass

    @abstractmethod
    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> int:
        pass

    @abstractmethod
    def free_block_hashes(self, request: Request) -> None:
        pass

    @abstractmethod
    def take_events(self) -> list[KVCacheEvent]:
        pass
