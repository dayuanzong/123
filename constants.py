import os
import re
from typing import Optional


# 环境变量的可能值

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"1"})


def _is_true(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.upper() in ENV_VARS_TRUE_VALUES


def _as_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    return int(value)


# Constants for file downloads 文件下载的常量

PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
CONFIG_NAME = "config.json"
REPOCARD_NAME = "README.md"

# Git-related constants

DEFAULT_REVISION = "main"
REGEX_COMMIT_OID = re.compile(r"[A-Fa-f0-9]{5,40}")

HUGGINGFACE_CO_URL_HOME = "https://huggingface.co/"

_staging_mode = _is_true(os.environ.get("HUGGINGFACE_CO_STAGING"))

ENDPOINT = os.getenv("HF_ENDPOINT") or ("https://hub-ci.huggingface.co" if _staging_mode else "https://huggingface.co")

HUGGINGFACE_CO_URL_TEMPLATE = ENDPOINT + "/{repo_id}/resolve/{revision}/{filename}"
HUGGINGFACE_HEADER_X_REPO_COMMIT = "X-Repo-Commit"
HUGGINGFACE_HEADER_X_LINKED_ETAG = "X-Linked-Etag"
HUGGINGFACE_HEADER_X_LINKED_SIZE = "X-Linked-Size"

INFERENCE_ENDPOINT = os.environ.get("HF_INFERENCE_ENDPOINT", "https://api-inference.huggingface.co")

REPO_ID_SEPARATOR = "--"
# ^ this substring is not allowed in repo_ids on hf.co	此子字符串不允许在hf.co的repo_ids中使用
# and is the canonical one we use for serialization of repo ids elsewhere. 是我们在其他地方序列化仓库id时使用的规范类型。


REPO_TYPE_DATASET = "dataset"
REPO_TYPE_SPACE = "space"
REPO_TYPE_MODEL = "model"
REPO_TYPES = [None, REPO_TYPE_MODEL, REPO_TYPE_DATASET, REPO_TYPE_SPACE]
SPACES_SDK_TYPES = ["gradio", "streamlit", "docker", "static"]

REPO_TYPES_URL_PREFIXES = {
    REPO_TYPE_DATASET: "datasets/",
    REPO_TYPE_SPACE: "spaces/",
}
REPO_TYPES_MAPPING = {
    "datasets": REPO_TYPE_DATASET,
    "spaces": REPO_TYPE_SPACE,
    "models": REPO_TYPE_MODEL,
}


# default cache
default_home = os.path.join(os.path.expanduser("~"), ".cache")
hf_cache_home = os.path.expanduser(
    os.getenv(
        "HF_HOME",
        os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "huggingface"),
    )
)

default_cache_path = os.path.join(hf_cache_home, "hub")
default_assets_cache_path = os.path.join(hf_cache_home, "assets")

HUGGINGFACE_HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE", default_cache_path)
HUGGINGFACE_ASSETS_CACHE = os.getenv("HUGGINGFACE_ASSETS_CACHE", default_assets_cache_path)

HF_HUB_OFFLINE = _is_true(os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"))

# Opt-out from telemetry requests
HF_HUB_DISABLE_TELEMETRY = _is_true(os.environ.get("HF_HUB_DISABLE_TELEMETRY") or os.environ.get("DISABLE_TELEMETRY"))

# In the past, token was stored in a hardcoded location  过去，token存储在硬编码的位置
# `_OLD_HF_TOKEN_PATH` is deprecated and will be removed "at some point".`_OLD_HF_TOKEN_PATH`已弃用，将在“某个时候”被移除。
# See https://github.com/huggingface/huggingface_hub/issues/1232
_OLD_HF_TOKEN_PATH = os.path.expanduser("~/.huggingface/token")
HF_TOKEN_PATH = os.path.join(hf_cache_home, "token")


# Here, `True` will disable progress bars globally without possibility of enabling it 这里，`True`将全局禁用进度条，无法启用它
# programmatically. `False` will enable them without possibility of disabling them. 以编程方式。`False`将启用它们，而不可能禁用它们。
# If environment variable is not set (None), then the user is free to enable/disable 如果没有设置环境变量(None)，那么用户可以自由启用/禁用
# them programmatically.
# TL;DR: env variable has priority over code TL;DR:环境变量优先于代码
__HF_HUB_DISABLE_PROGRESS_BARS = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
HF_HUB_DISABLE_PROGRESS_BARS: Optional[bool] = (
    _is_true(__HF_HUB_DISABLE_PROGRESS_BARS) if __HF_HUB_DISABLE_PROGRESS_BARS is not None else None
)

# Disable warning on machines that do not support symlinks (e.g. Windows non-developer) 在不支持符号链接的机器上禁用警告(例如Windows非开发人员)
HF_HUB_DISABLE_SYMLINKS_WARNING: bool = _is_true(os.environ.get("HF_HUB_DISABLE_SYMLINKS_WARNING"))

# Disable warning when using experimental features 在使用实验特性时禁用警告
HF_HUB_DISABLE_EXPERIMENTAL_WARNING: bool = _is_true(os.environ.get("HF_HUB_DISABLE_EXPERIMENTAL_WARNING"))

# Disable sending the cached token by default is all HTTP requests to the Hub 默认情况下，发送缓存令牌是所有到集线器的HTTP请求
HF_HUB_DISABLE_IMPLICIT_TOKEN: bool = _is_true(os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN"))

# Enable fast-download using external dependency "hf_transfer" 使用外部依赖项“hf_transfer”支持快速下载
# See:
# - https://pypi.org/project/hf-transfer/
# - https://github.com/huggingface/hf_transfer (private)
HF_HUB_ENABLE_HF_TRANSFER: bool = _is_true(os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"))


# Used if download to `local_dir` and `local_dir_use_symlinks="auto"`用于下载到`local_dir`和`local_dir_use_symlinks="auto"`
# Files smaller than 5MB are copy-pasted while bigger files are symlinked. The idea is to save disk-usage by symlinking	小于5MB的文件采用复制粘贴，大于5MB的文件采用符号链接。其思想是通过符号链接来节省磁盘使用量
# huge files (i.e. LFS files most of the time) while allowing small files to be manually edited in local folder.大型文件(大多数情况下是LFS文件)，同时允许在本地文件夹手动编辑小文件。
HF_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD: int = (
    _as_int(os.environ.get("HF_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD")) or 5 * 1024 * 1024
)
