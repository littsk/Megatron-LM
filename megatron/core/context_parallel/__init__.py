from typing import Literal, Optional

from .backend import (
    ContextParallelHandler,
    DefaultContextParallelHandler,
    MagiAttnContextParallelHandler,
    TEDynamicContextParallelHandler,
)


def get_cp_handler_cls(
    transformer_backend: Optional[
        Literal["transformer_engine", "local", "inference_optimized"]
    ] = None,
    context_parallel_backend: Optional[Literal["default", "magi", "dynamic_cp"]] = None,
) -> type[ContextParallelHandler]:
    """
    Factory function to get the appropriate Context Parallel Handler class based on the backend.
    Be careful to select the appropriate transformer_backend and context_parallel_backend.
    Using `transformer_engine` impl, the supported handlers are `default` and `dynamic_cp`.
    Using `local` impl, the supported handlers are `default` and `magi`.

    Args:
        transformer_backend: Specifies which transformer implementation to use
            ('transformer_engine', 'local', or 'inference_optimized').
        context_parallel_backend: Specifies which context-parallel handler to use
            ('default', 'magi', or 'dynamic_cp').

    Returns:
        The class definition of the appropriate ContextParallelHandler.

    Raises:
        ValueError: If an unsupported handler is provided.
    """
    if transformer_backend == "transformer_engine":
        assert context_parallel_backend in (
            "default",
            "dynamic_cp",
        ), "transformer_engine supports only the 'default' and 'dynamic_cp' handlers."
    elif transformer_backend == "local":
        assert context_parallel_backend in (
            "default",
            "magi",
        ), "local supports only the 'default' and 'magi' handlers."
    elif transformer_backend == "inference_optimized":
        assert context_parallel_backend in (
            "default"
        ), "transformer_engine supports only the 'default' handler."
    else:
        raise ValueError(f"Unsupported transformer impl backend, got {transformer_backend}.")
    if context_parallel_backend == "default":
        return DefaultContextParallelHandler
    elif context_parallel_backend == "magi":
        return MagiAttnContextParallelHandler
    elif context_parallel_backend == "dynamic_cp":
        return TEDynamicContextParallelHandler
    else:
        raise ValueError(f"Unsupported context_parallel_backend, got {context_parallel_backend}.")
