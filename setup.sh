sudo pip uninstall jax jaxlib libtpu-nightly libtpu libtpu_-tpuv4 -y
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install smart_open[gcs] gin-config func_timeout optax==0.0.6 ray dm-haiku einops transformers
export PYTHONPATH=$PYTHONPATH:$PWD