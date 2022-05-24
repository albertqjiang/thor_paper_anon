# Subgoal Search ATP

Setup required to run jobs:
* Create a venv
* Inside this venv install mrunner:
  * `git clone https://gitlab.com/awarelab/mrunner.git && cd mrunner && git checkout gcp_backend`
  * `pip install -e .`
* Go to the root folder of this repository (`atp`) and install the remaining requirements:
  * `pip install -r requirements.txt`
* Copy `mrunner_config_example.yaml` to `mrunner_example.yaml` and set `vm_name` and `zone` according to your TPU VM names/location.

## Run jobs on TPU

`mrunner --config=mrunner_config.yaml --context=tpu run experiments/transformer_finetune.py`

## Run locally

`python3 runner.py --mrunner --config experiments/transformer_local.py`

## Setup Isabelle

Log onto the virtual machine and execute copy_isabelle_data.sh

Install zip:

```shell
sudo apt install zip
```

Install SDKMAN 

```shell
curl -s "https://get.sdkman.io" | bash
source .bashrc
 ```
Try
```shell
sdk help
```
to makes ure sdk is properly installed.
    
Install JAVA 11 and sbt
```shell
sdk install java 11.0.11-open
sdk install sbt
```

## Troubleshooting

If you get permission error when you code tries to access your bucket, type the following command on VM:

`gcloud auth application-default login`

If you have problem with installing T5X and see error:
`error: invalid command 'bdist_wheel'`, try:

`pip install wheel`
`python setup.py bdist_wheel`

If locally JAX is not using GPU (and you have CUDA installed):

``pip install --upgrade jaxlib==0.1.75+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_releases.html``

If you get sbt java: command not found error:

``sudo apt-get install default-jre``