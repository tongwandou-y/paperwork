import importlib.util
import inspect
import os
from pathlib import Path

from torch import nn


def _load_module_from_file(module_file_path: Path):
    module_name = f"dynamic_model_{module_file_path.stem.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, str(module_file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从文件加载模块: {module_file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_first_model_class(module):
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, nn.Module) and obj is not nn.Module:
            return obj
    return None


def build_model_from_config(configs):
    model_dir = Path(__file__).resolve().parents[1] / "models" / "model"
    model_file = getattr(configs, "model_file", None)
    if not model_file:
        model_type = getattr(configs, "model_type", "DNN")
        model_file = f"{model_type}.py"

    model_file = str(model_file)
    if not model_file.endswith(".py"):
        model_file = f"{model_file}.py"

    model_path = model_dir / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    module = _load_module_from_file(model_path)
    model_class_name = getattr(configs, "model_class", None)

    model_cls = None
    if model_class_name:
        if not hasattr(module, model_class_name):
            raise AttributeError(f"模型文件 '{model_file}' 中未找到类 '{model_class_name}'")
        model_cls = getattr(module, model_class_name)
    else:
        model_cls = _find_first_model_class(module)
        if model_cls is None:
            raise ValueError(f"模型文件 '{model_file}' 中未找到 nn.Module 子类")

    if not issubclass(model_cls, nn.Module):
        raise TypeError(f"类 '{model_cls.__name__}' 不是 nn.Module 子类")

    # 保证用于命名的 model_type 与实际文件一致（去后缀）
    configs.model_type = os.path.splitext(os.path.basename(model_file))[0]
    return model_cls(configs), model_cls.__name__
