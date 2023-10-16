import os
import pathlib
import shutil

import transformers
import random
import torch
import numpy as np


def fix_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


class CheckpointCallback(transformers.TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        """Add complete indicator to avoid incomplete checkpoints."""
        if state.is_world_process_zero:
            ckpt_path = os.path.join(args.output_dir,
                                     f'checkpoint-{state.global_step}')
            with open(os.path.join(ckpt_path, 'complete'), 'w') as f:
                f.write('')
            print(f'Checkpoint {state.global_step} saved.')


def cleanup_incomplete_checkpoints(output_dir):
    """Remove incomplete checkpoints.
    
    Check the existence of checkpoints in all processes.
    All ranks must simultaneously resume from a checkpoint if it exists.
    Otherwise, upon recovery the model weights may not reload correctly,
    causing loss spikes.
    """
    checkpoints = list(pathlib.Path(output_dir).glob('checkpoint-*'))
    checkpoints = [c for c in checkpoints if c.name.split('-')[-1].isdigit()]
    checkpoints = sorted(checkpoints,
                         key=lambda x: int(x.name.split('-')[-1]),
                         reverse=True)
    for checkpoint in checkpoints:
        if not (checkpoint / 'complete').exists():
            print(f'Removing incomplete checkpoint {checkpoint}')
            shutil.rmtree(checkpoint)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
