"""Vendored subset of DeepTangle (kirkegaardlab/deeptangle, MIT) as used by Tierpsy Tracker 2.0 / DeepTangleCrawl.
Only model + inference: build_model, non_max_suppression. Dataset / tracking modules are not included.
"""
from .forward import build_model
from .predict import Predictions, non_max_suppression
