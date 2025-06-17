"""Enum классы для описания граничных условий."""
from enum import Enum


class TypeBC(Enum):
	Dirichlet: str
	Neyman: str

class Bound(Enum):
	Left: str
	Right: str
	Top: str
	Bottom: str
