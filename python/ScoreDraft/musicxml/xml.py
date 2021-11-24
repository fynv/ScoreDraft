from enum import Enum

__NAMESPACE__ = "http://www.w3.org/XML/1998/namespace"


class LangValue(Enum):
    VALUE = ""


class SpaceValue(Enum):
    DEFAULT = "default"
    PRESERVE = "preserve"
