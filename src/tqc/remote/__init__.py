"""Remote execution, cloud orchestration, and packaging for TQC benchmarks."""

from .kaggle_auth import (
    KaggleAuthError,
    ensure_kaggle_cli_installed,
    resolve_kaggle_credentials,
    validate_kaggle_credentials,
)
from .kaggle_packager import (
    KagglePackagingError,
    build_kernel_metadata,
    create_remote_entrypoint,
    package_kernel_directory,
    run_local_preflight_smoke,
)
from .queue_manager import (
    DualSlotQueueManager,
    QueueError,
    RemoteJob,
)

__all__ = [
    "KaggleAuthError",
    "resolve_kaggle_credentials",
    "validate_kaggle_credentials",
    "ensure_kaggle_cli_installed",
    "KagglePackagingError",
    "build_kernel_metadata",
    "create_remote_entrypoint",
    "package_kernel_directory",
    "run_local_preflight_smoke",
    "DualSlotQueueManager",
    "QueueError",
    "RemoteJob",
]
