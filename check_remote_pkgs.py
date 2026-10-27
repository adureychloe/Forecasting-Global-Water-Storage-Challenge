import importlib

for name in ("catboost", "xgboost", "pandas", "numpy"):
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "?"))
