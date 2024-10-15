from dp.launching.cli import (
    SubParser,
    run_sp_and_exit,
    default_minimal_exception_handler
)

from app.dist_model import Dist
from app.dist_runner import DistRunner
from app.ft_model import Finetune
from app.ft_runner import FinetuneRunner

def to_parser():
    return {
        "Model Distillation": SubParser(Dist, DistRunner, "Running model distillation"),
        "Model Finetune": SubParser(Finetune,FinetuneRunner,"Running model finetune")
    }

if __name__ == "__main__":
    run_sp_and_exit(
        to_parser(),
        description="App demo for pfd: fast finetune and distillation from pretrained atomic model",
        version="0.0.0",
        exception_handler=default_minimal_exception_handler
    )