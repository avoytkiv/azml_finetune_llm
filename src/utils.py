import os
import pathlib
import shutil

import transformers

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
    """Remove incomplete checkpoints."""
    checkpoints = list(pathlib.Path(output_dir).glob('checkpoint-*'))
    checkpoints = [c for c in checkpoints if c.name.split('-')[-1].isdigit()]
    checkpoints = sorted(checkpoints,
                         key=lambda x: int(x.name.split('-')[-1]),
                         reverse=True)
    for checkpoint in checkpoints:
        if not (checkpoint / 'complete').exists():
            print(f'Removing incomplete checkpoint {checkpoint}')
            shutil.rmtree(checkpoint)