# Fine-Tuning LLM with SkyPilot and DVC   

<img width="1057" alt="Screenshot 2023-10-19 at 15 51 20" src="https://github.com/avoytkiv/azml_finetune_llm/assets/74664634/752023ad-12a1-4a94-9f55-392a415c341e">   


In this project, I fine-tune the `bert-base-uncased` model for text classification on the `hotels-reviews` dataset.  

There are a few learning goals for this project:  

- **Provisioning/Infrastructure**: Run the training pipeline in the cloud on a GPU instance in the most efficient way across multiple cloud providers (cost, performance, checkpointing, spot instances, etc.).
- **Machine Learning**: How fine-tuning improves the performance of the model.
- **MLOps**: Compare ML experiments on Weights & Biases vs DVC Studio - best tool, advanteages and disadvantages.

Tools used in this project:

- HuggingFace Transformers for fine-tuning the model.
- DVC for defining machine learning pipelines - dependencies.
- SkyPilot for provisioning infrastructure and running the training pipeline in the cloud.
- Weights & Biases for logging metrics and artifacts.

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

install SkyPilot and DVC using pip


```shell
pip install "skypilot[all]"
```

Next, configure AWS cloud:

```
pip install boto3
aws configure
```

Confirm the setup with the following command:
```shell
sky check
```

After configuring the setup, download the data to your local machine, then upload it to your own bucket (where you have write access).

```shell
# Pull data from remote storage to local machine
$ dvc pull
# Configure your own bucket in .dvc/config:
#   - AWS: https://iterative.ai/blog/aws-remotes-in-dvc
#   - GCP: https://iterative.ai/blog/using-gcp-remotes-in-dvc
#   - Azure: https://iterative.ai/blog/azure-remotes-in-dvc
# Push the data to your own bucket
$ dvc push
```

Usually current remote URL for origin is using HTTPS. If you want to use SSH keys for authentication, you should change this URL to the SSH format. You can do this with:

```shell
git remote set-url origin git@github.com:avoytkiv/azml_finetune_llm.git
```

Also, check permissions for the SSH key and change them if needed. This error may occur if the permissions are not correct:

>[!ERROR]   
>The remote server unexpectedly closed the connection.owner or permissions on /home/ubuntu/.ssh/config

This can be fixed by changing the permissions of the config file:

```shell
chmod 600 ~/.ssh/config
```

More details can be found [here](https://serverfault.com/questions/253313/ssh-returns-bad-owner-or-permissions-on-ssh-config).


## SkyPilot: Run everything in Cloud

Submit a run job to the cloud and pull the results to your local machine.

To launch a cloud instance and submit job, run:

```shell
sky launch -c vscode -i 60 sky-vscode.yaml
```

To launch on spot instances, run:

```shell
sky launch sky-vscode.yaml  -c vscode -d --use-spot
```

To launch explicitly on AWS, run:

```shell
sky gpunode --cloud aws --instance-type g4dn.2xlarge --region us-west-1 --cpus 8
```

The skyline command will launch a VS Code tunnel to the cloud instance. Once the tunnel is created, you can open the VS Code instance in your browser by clicking the link in the terminal output.

When you are ready to launch a long-running training job, run:

```shell
sky launch -c train --use-spot -i 30 --down sky-training.yaml
```

This SkyPilot command uses spot instances to save costs and automatically terminates the instance after 30 minutes of idleness. Once the experiment is complete, its artifacts such as model weights and metrics are stored in your bucket (thanks to the dvc exp push origin command in sky-training.yaml).

Add `--env DVC-STUDIO-TOKEN` to `sky launch/exec` command to see the experiment running live in DVC Studio.
Add `--env WANDB_API_KEY` to `sky launch/exec` command to see the experiment running live in Weights & Biases.
First, make it available in your current shell.

While the model is training you can monitor the logs by running the following command.

```shell
sky logs train
```

Then, you can pull the results of the experiment to your local machine by running:
    
```shell
dvc exp pull origin
```

You can change the cloud provider and instance type in the resources section of sky-training.yaml or sky-vscode.yaml.

In the YAML’s file_mounts section, we specified that a bucket named $ARTIFACT_BUCKET_NAME (passed in via an env var) should be mounted at /artifacts inside the VM:

```shell
file_mounts:
  /artifacts:
    name: $ARTIFACT_BUCKET_NAME
    mode: MOUNT
```
When launching the job, we then simply pass `/artifacts` to its `--output_dir` flag, to which it will write all checkpoints and other artifacts

In other words, your training program uses this mounted path as if it’s local to the VM! Files/dirs written to the mounted directory are automatically synced to the cloud bucket.

>[!NOTE]  
>There’s one edge case to handle, however: During a checkpoint write, the instance may get preempted suddenly and only partial
>state is written to the cloud bucket. When this happens, resuming from a corrupted partial checkpoint will crash the program.

## Data Science Workflow

1. Fine-tune the `bert-base-uncased` model for text classification on the `hotels-reviews` dataset.
2. Evaluate the model on the `hotels-reviews-small` dataset.
3. Use DVC to track metrics, model, and parameters across the train and evaluate stages.

## Usefull Commands

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
- [Request quota increase](https://skypilot.readthedocs.io/en/latest/cloud-setup/quota.html#quota)
- [Azure - GPU optimized virtual machine sizes](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu)
- [DVC Documentation](https://dvc.org/doc)
- [ML experiments in the cloud with SkyPilot and DVC](https://alex000kim.com/tech/2023-08-10-ml-experiments-in-cloud-skypilot-dvc/)
- [Fine-Tuning Large Language Models with a Production-Grade Pipeline](https://iterative.ai/blog/finetune-llm-pipeline-dvc-skypilot)
- [Skypilot LLM](https://github.com/skypilot-org/skypilot/tree/master/llm)
- [Create SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
- [WANDB - Logging with Weights and Biases in Transformer](https://docs.wandb.ai/guides/integrations/huggingface)

