from .backprop import SGDRule
from .base import LearningRule
from .modulated import PerLayerLRRule
from .tasks import TASK_REGISTRY

RULE_REGISTRY = {
    "sgd": SGDRule,
    "per_layer_lr": PerLayerLRRule,
}
