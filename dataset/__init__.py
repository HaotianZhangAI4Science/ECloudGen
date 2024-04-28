DATASETS_REGISTRY = {}

def build_datasets(cfg, task, mode):
    print('Build {} datasets from {}'.format(cfg.config, DATASETS_REGISTRY[cfg.config]))
    return DATASETS_REGISTRY[cfg.config].build_datasets(cfg, task, mode)

def register_datasets(names):
    """
    New model types can be added to unicore with the :func:`register_model`
    function decorator.

    For example::

        @register_model("lstm")
        class LSTM(UnicoreEncoderDecoderModel):
            (...)

    Args:
        name (str): the name of the model
    """

    def register_datasets_cls(cls):
        for name in names:
            if name in DATASETS_REGISTRY:
                raise ValueError("Cannot register duplicate datasets ({})".format(name))
            DATASETS_REGISTRY[name] = cls
        return cls

    return register_datasets_cls
