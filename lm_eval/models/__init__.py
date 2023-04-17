from . import gpt2
from . import gpt3
from . import textsynth
from . import dummy
from . import opt
from . import pythia
from . import cerebras

MODEL_REGISTRY = {
    "hf": gpt2.HFLM,
    "gpt2": gpt2.GPT2LM,
    "gpt3": gpt3.GPT3LM,
    'opt': opt.OPT,
    'pythia': pythia.Pythia,
    'cerebras': cerebras.Cerebras,
    "textsynth": textsynth.TextSynthLM,
    "dummy": dummy.DummyLM,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
