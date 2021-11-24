from enum import Enum

__NAMESPACE__ = "http://www.w3.org/1999/xlink"


class ActuateValue(Enum):
    ON_REQUEST = "onRequest"
    ON_LOAD = "onLoad"
    OTHER = "other"
    NONE = "none"


class ShowValue(Enum):
    NEW = "new"
    REPLACE = "replace"
    EMBED = "embed"
    OTHER = "other"
    NONE = "none"


class TypeValue(Enum):
    SIMPLE = "simple"
