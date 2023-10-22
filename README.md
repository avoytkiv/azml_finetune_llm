# Fine-Tuning LLM with SkyPilot and DVC   

<img width="1057" alt="Screenshot 2023-10-19 at 15 51 20" src="https://github.com/avoytkiv/azml_finetune_llm/assets/74664634/752023ad-12a1-4a94-9f55-392a415c341e">   


In this project, I fine-tune the `bert-base-uncased` model for text classification on the `hotels-reviews` dataset. 
The dataset is artificially made and contains 100k reviews whith labels: `Excellent`, `Very good`, `Average`, `Poor`, `Terrible`. Spoiler alert: the model is able to learn the task and achieve 100% accuracy with no more than 200 samples.

There are a few *learning goals* for this project:  

- **Provisioning/Infrastructure**: Run the training pipeline in the cloud on a GPU instance in the most efficient way across multiple cloud providers (cost, performance, checkpointing, spot instances, etc.).
- **Machine Learning**: How fine-tuning improves the performance of the model.
- **MLOps**: Compare ML experiments on Weights & Biases vs DVC Studio - best tool, advanteages and disadvantages.

*Tools* used in this project:

- `HuggingFace Transformers` for fine-tuning the model.
- `DVC` for defining machine learning pipelines - dependencies.
- `SkyPilot` for provisioning infrastructure and running the training pipeline in the cloud.
- `Weights & Biases` for logging metrics and artifacts.

**Tasks**
- [x] Preprocess the custom `hotels-reviews` dataset.
    - [x] Convert the dataset to the HuggingFace format.
    - [x] Split the dataset into train and test sets.
    - [x] Tokenize the dataset.
- [x] Evaluate the `bert-base-uncased` model on the preprocessed dataset.
- [x] Fine-tune the `bert-base-uncased` model.
- [x] Set up infrastructure to run the training pipeline in the cloud on a GPU instance.
    - [x] Register a new account on: 
        - Lambda (AI cloud platform), 
        - Cloudflare (R2 storage with zero egress charges), 
        - AWS, Azure, GCP (major GPU cloud providers) 
    - [x] Request quota increase for GPU instances.
    - [x] Install SkyPilot, DVC, and Weight & Biases.
    - [x] Authenticate with AWS, Azure, GCP etc. Skypilot will choose the cloud provider based on GPU availability and pricing.
    - [x] Upload the data to S3 (tracked by DVC).
    - [x] Create a SkyPilot configuration to run the training job in the cloud.
        - [x] Configure resources (cloud provider, instance type, etc.).
        - [x] Configure file mounts (artifacts, data, etc.)
        - [x] Configure the training job (command, environment variables, etc.).
    - [x] Create SSH keys to connect to GitHub (DVC needs it as it works with Git).
    - [x] Implement checkpoints to save the model weights and metrics to WandB. It will allow to resume training from the last checkpoint. With this setup we can run the training job for a long time on spot instances (`sky spot launch`) and get automatic recovery from preemption.
    - [x] Implement `Early Stopping` to not waste time on training the model that is not improving.

    Bonus tasks:
    - [ ] Benchmark performance and cost of different GPU instances on different cloud providers (`sky bench launch`).
        - [ ] Make a table with the results.
    - [ ] Check Sky Spot Instances Dashboard


## Setup

Install SkyPilot, DVC, and Weight & Biases.

```shell
pip install requirements.txt
```

Next, configure AWS, Azure, GCP, etc. credentials. SkyPilot will choose the cloud provider based on GPU availability and pricing.

Example of AWS configuration:

```
pip install boto3
aws configure
```

Confirm the setup with the following command:
```shell
sky check
```

Define the *resources, file mounts, setup and command for the training job* in the SkyPilot configuration file `sky-vscode.yaml`. 

File mounts are used to mount the data, ssh keys and gitconfig to the cloud instance. The least two are needed for DVC to work with Git. 

```yaml
file_mounts:
  /data: ~/azml_finetune_llm/data
  ~/.ssh/id_ed25519: ~/.ssh/id_ed25519
  ~/.ssh/id_ed25519.pub: ~/.ssh/id_ed25519.pub
  ~/.gitconfig: ~/.gitconfig
```

Setup is running only once when the instance is created. It is used to install dependencies.

Finally, set the commands to run the training job. SkyPilot creates a new working directory `sky_workdir`, so we need to change the directory to the project root. Then we can run the ML pipeline with one command thanks to DVC.

```yaml
run: |
  cd ~/sky_workdir
  source activate pytorch
  dvc exp run 
```

>[!NOTE]
>Usually current remote URL for origin is using HTTPS. If you want to use SSH
keys for authentication, you should change this URL to the SSH format. You can do this with:
```shell
git remote set-url origin git@github.com:avoytkiv/azml_finetune_llm.git
```


Also, check permissions for the SSH key and change them if needed. This error may occur if the permissions are not correct:

>[!Warning]   
>The remote server unexpectedly closed the connection.owner or permissions on /home/ubuntu/.ssh/config

This can be fixed by changing the permissions of the config file:

```shell
chmod 600 ~/.ssh/config
```

More details can be found [here](https://serverfault.com/questions/253313/ssh-returns-bad-owner-or-permissions-on-ssh-config).


## SkyPilot: Run everything in Cloud

To launch job on spot instances, run:

```shell
sky launch sky-vscode.yaml -c mycluster -i 30 -d --use-spot
```

This SkyPilot command uses spot instances to save costs and automatically terminates the instance after 30 minutes of idleness. Once the experiment is complete, its artifacts such as model weights and metrics are logged to Weights & Biases.

Add `--env DVC-STUDIO-TOKEN` to `sky launch/exec` command to see the experiment running live in DVC Studio.
Add `--env WANDB_API_KEY` to `sky launch/exec` command to see the experiment running live in Weights & Biases.
First, make it available in your current shell.

While the model is training, you can monitor the logs by running the following command.

```shell
sky logs mycluster
```

## Checkpoints

HuggingFace Transformers supports checkpointing. And has an integration with Weights & Biases. To enable checkpointing, we need to: 
- set the environment variable `WANDB_LOG_MODEL=checkpoint`.
- set `--run_name` to `$SKYPILOT_TASK_ID` so that the logs for all recoveries of the same job will be saved to the same run in Weights & Biases.

Any Transformers Trainer you initialize from now on will upload models to your W&B project. Model checkpoints will be logged and include the full model lineage.  

<img width="50%" alt="Screenshot 2023-10-21 at 16 04 10" src="https://github.com/avoytkiv/azml_finetune_llm/assets/74664634/f65cf0ad-a0e4-417f-b5bb-88d4c1c2cd52"> <img width="50%" alt="Screenshot 2023-10-21 at 15 24 35" src="https://github.com/avoytkiv/azml_finetune_llm/assets/74664634/05e418ca-9b07-4b69-8e2a-4dfb8729d4ed">


Any time the instance is preempted (interrupted), the SkyPilot will automatically resume the training job from the last checkpoint.

>[!NOTE]  
>There’s one edge case to handle: during a checkpoint write, the instance may get preempted suddenly and only partial
>state is written to the cloud bucket. When this happens, resuming from a corrupted partial checkpoint will crash the program. The `cleanup_incomplete_checkpoints` function will delete any partial checkpoints that are incomplete.


## Data Science Workflow

1. Evaluate the `bert-base-uncased` model on the `hotels-reviews-small` dataset for baseline performance (it's 20% accuracy).
2. Fine-tune the `bert-base-uncased` model for text classification on the `hotels-reviews` dataset.
3. Evaluate the model on the `hotels-reviews-small` dataset.
4. Use WandB to track metrics, model, and parameters across the train and evaluate stages.


## Results

Now, when the ml pipeline is defined and the cloud infrastructure is optimized for cost, we can run and then compare our experiments. Not only `train` and `evaluate` stages, but also `system` metrics such as GPU utilization, memory usage, etc. are logged to Weights & Biases.

![wandb](https://github.com/avoytkiv/azml_finetune_llm/assets/74664634/4ccb6449-09bc-4342-a2bf-43c4e70db523) 

The model is able to learn the task and achieve 100% accuracy with no more than 200 samples.

## What's next

Use the Weights & Biases Model Registry to register models to prepare it for `staging` or `deployment` in production environment.


## Useful Commands

Freeze only the packages that are required to run the project.

```shell
pip freeze -q -r requirements.txt | sed '/freeze/,$ d' > requirements-froze.txt
mv requirements-froze.txt requirements.txt
``` 


## Useful Resources
- [SkyPilot Documentation](https://skypilot-dev.readthedocs.io/en/latest/)
- [SkyPilot - Configure access to cloud providers](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html)
- [SkyPilot - Source code for sky.Task - debugging](https://sky-proj-sky.readthedocs-hosted.com/en/latest/_modules/sky/task.html)
- [SkyPilot - SkyCallback](https://skypilot.readthedocs.io/en/latest/reference/benchmark/callback.html#integrations-with-ml-frameworks)
- [Skypilot - LLM](https://github.com/skypilot-org/skypilot/tree/master/llm)
- [SkyPilot - Request quota increase](https://skypilot.readthedocs.io/en/latest/cloud-setup/quota.html#quota)
- [Azure - GPU optimized virtual machine sizes](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu)
- [DVC Documentation](https://dvc.org/doc)
- [ML experiments in the cloud with SkyPilot and DVC](https://alex000kim.com/tech/2023-08-10-ml-experiments-in-cloud-skypilot-dvc/)
- [Fine-Tuning Large Language Models with a Production-Grade Pipeline](https://iterative.ai/blog/finetune-llm-pipeline-dvc-skypilot)
- [Create SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
- [WANDB - Logging with Weights and Biases in Transformer](https://docs.wandb.ai/guides/integrations/huggingface)

