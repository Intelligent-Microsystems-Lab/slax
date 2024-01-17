# Slax

Spike-based learning in Flax/JAX.

Slax is a spiking neural network package, focusing on easy and efficient implementations of many training algorithms, especially online training. We will also branch into reinforcement learning.

While the repository is private, install by cloning the repository, navigating to the local folder, and running `pip install .`

## TO DO

* Minor items
  * Make OSTL and algs work with any number of carry states (only works with one now). This is necessary for general compatibility
  * Add more neurons and surrogate derivatives (Currently only LIF and fast_sigmoid are fully operational)
  * Make `offline_learning` compatible with batch normalization
  * Determine best output for `gen_loss_landscape`
  * Add more info to function comments, especially function outputs
  * Code cleanup
* Major items
  * Add docs website
  * Add package to PyPI