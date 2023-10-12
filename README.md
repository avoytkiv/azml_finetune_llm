# Fine-Tuning LLM with SkyPilot and DVC

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


## SkyPilot: Run everything in Cloud

Submit a run job to the cloud and pull the results to your local machine.

To launch a cloud instance for interactive development, run:

```shell
sky launch -c vscode -i 60 sky-vscode.yaml
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

While the model is training you can monitor the logs by running the following command.

```shell
sky logs train
```

Then, you can pull the results of the experiment to your local machine by running:
    
```shell
dvc exp pull origin
```

You can change the cloud provider and instance type in the resources section of sky-training.yaml or sky-vscode.yaml.

## Data Science Workflow

1. Fine-tune the `bert-base-uncased` model for text classification on the `hotels-reviews` dataset.
2. Evaluate the model on the `hotels-reviews-small` dataset.
3. Use DVC to track metrics, model, and parameters across the train and evaluate stages.

## Useful Resources
-[ML experiments in the cloud with SkyPilot and DVC](https://alex000kim.com/tech/2023-08-10-ml-experiments-in-cloud-skypilot-dvc/)
-[Fine-Tuning Large Language Models with a Production-Grade Pipeline](https://iterative.ai/blog/finetune-llm-pipeline-dvc-skypilot)

