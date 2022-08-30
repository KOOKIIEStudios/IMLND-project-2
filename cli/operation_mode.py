"""This module represents the possible modes of operation"""
from enum import Enum


class Mode(Enum):
    """This class models the modes of operation"""
    NO_ARGS: int = 1
    TOP_K: int = 2
    CATEGORY: int = 3
    BOTH: int = 4


def get_mode(has_k_flag: bool, has_category_flag: bool) -> int:
    if has_k_flag and has_category_flag:
        return Mode.BOTH
    if has_k_flag:
        return Mode.TOP_K
    if has_category_flag:
        return Mode.CATEGORY
    return Mode.NO_ARGS
