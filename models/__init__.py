MODEL_REGISTRY = {}

def build_model(cfg, task):
    print('Build {} model from {}'.format(cfg.config, MODEL_REGISTRY[cfg.config]))
    return MODEL_REGISTRY[cfg.config].build_model(cfg, task)

def register_model(names):
    """
    New model types can be added to unicore with the :func:`register_model`
    function decorator.

    For example::

        @register_model(["lstm"])
        class LSTM(UnicoreEncoderDecoderModel):
            (...)

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        for name in names:
            if name in MODEL_REGISTRY:
                raise ValueError("Cannot register duplicate model ({})".format(name))
            MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls
