import jax
import jax.numpy as jnp

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def gen_loss_landscape(get_loss,load_params,n_iter,n_models=1):
  '''
  Generates a loss landscape plot from saved saved parameters throughout training.
  
  Args:
  get_loss: A function that takes in the model parameters and outputs a scalar loss
  load_params: A function that takes which number to load and outputs the parameters in a list
  n_iter: number of saved parameter checkpoints
  n_models: number of models to plot on the loss landscape
  '''
  model_indicator = 0


  def npvec_to_tensorlist(pc, params):
    tree_val, tree_struct = jax.tree_util.tree_flatten(params)
    val_list = []
    counter = 0
    for i in [x.shape for x in tree_val]:
      increase = np.prod(i)
      val_list.append(pc[counter: int(counter + increase)].reshape(i))
      counter += increase

    return jax.tree_util.tree_unflatten(tree_struct, val_list)


  def project1d(w, d):
    assert len(w) == len(d), "dimension does not match for w and "
    return jnp.dot(w, d) / np.linalg.norm(d)


  def project2d(d, dx, dy, proj_method):
    if proj_method == "cos":
      # when dx and dy are orthorgonal
      x = project1d(d, dx)
      y = project1d(d, dy)
    elif proj_method == "lstsq":
      A = np.vstack([dx, dy]).T
      [x, y] = np.linalg.lstsq(A, d)[0]

    return x, y


  def get_surface(x, y, xdirection, ydirection, variables):

    xv, yv = jnp.meshgrid(x, y)

    def surface_parallel(ix, iy):
      interpolate_vars = jax.tree_util.tree_map(
          lambda w, x, y: w + x * ix + y * iy,
          variables,
          xdirection,
          ydirection,
      )
      return get_loss(interpolate_vars)


    zv_list = []
    for i in range(int(xv.flatten().shape[0] / 100)):
      zv = jax.vmap(surface_parallel)(
          jnp.array(xv.flatten())[(i * 100): (i + 1) * 100],
          jnp.array(yv.flatten())[(i * 100): (i + 1) * 100],
      )
      zv_list.append(zv)

    return xv, yv, np.stack(zv_list).flatten().reshape(xv.shape)


  params_end = load_params(n_iter-1)

  matrix = []
  for i in range(n_iter):
    for j in range(n_models):
      tmp = load_params(i)
      diff_tmp = jax.tree_map(lambda x, y: x - y, tmp[j], params_end[model_indicator])
      matrix.append(jnp.hstack([x.reshape(-1)
                    for x in jax.tree_util.tree_flatten(diff_tmp)[0]]))


  pca = PCA(n_components=2)
  pca.fit(np.array(matrix))

  pc1 = np.array(pca.components_[0])
  pc2 = np.array(pca.components_[1])

  angle = jnp.dot(pc1, pc2) / (jnp.linalg.norm(pc1) * jnp.linalg.norm(pc2))

  xdirection = npvec_to_tensorlist(pc1, params_end[model_indicator])
  ydirection = npvec_to_tensorlist(pc2, params_end[model_indicator])

  ratio_x = pca.explained_variance_ratio_[0]
  ratio_y = pca.explained_variance_ratio_[1]

  dx = pc1
  dy = pc2

  xcoord = {}
  ycoord = {}
  x_abs_max = 0
  y_abs_max = 0
  for j in range(n_models):
    xcoord[j] = []
    ycoord[j] = []
    for i in range(n_iter):
      tmp = load_params(i)
      diff_tmp = jax.tree_map(lambda x, y: x - y, tmp[j], params_end[model_indicator])
      diff_tmp = jnp.hstack([x.reshape(-1)
                            for x in jax.tree_util.tree_flatten(diff_tmp)[0]])

      tmp_x, tmp_y = project2d(diff_tmp, dx, dy, 'cos')
      xcoord[j].append(tmp_x)
      ycoord[j].append(tmp_y)

      if np.abs(tmp_x) > x_abs_max:
        x_abs_max = abs(tmp_x)
      if np.abs(tmp_y) > y_abs_max:
        y_abs_max = abs(tmp_y)


  buffer_y = y_abs_max * 0.05
  buffer_x = x_abs_max * 0.05

  x = np.linspace(
      (-1*x_abs_max) - buffer_x,
      x_abs_max + buffer_x,
      100,
  )
  y = np.linspace(
      (-1*y_abs_max) - buffer_y,
      y_abs_max + buffer_y,
      100,
  )


  xv, yv, zv = get_surface(x, y, xdirection, ydirection, params_end[model_indicator])

  font_size = 23
  gen_lw = 8

  plt.rc("font", weight="bold")
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14.4, 8.5))


  def fmt(x):
    s = f"{x:.3f}"
    if s.endswith("0"):
      s = f"{x:.0f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"



  CS = ax.contour(xv, yv, zv, 100)
  ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)

  for j in range(n_models):
    ax.plot(
        xcoord[j],
        ycoord[j],
        label=str(j),
        #color='red',
        marker="o",
        markeredgecolor='black',
        markerfacecolor="None",
        markersize=8,
        linewidth=gen_lw,
    )

  ax.set_xlabel(
      "1st PC: %.2f %%" % (ratio_x * 100),
      fontdict={"weight": "bold", "size": font_size},
  )
  ax.set_ylabel(
      "2nd PC: %.2f %%" % (ratio_y * 100),
      fontdict={"weight": "bold", "size": font_size},
  )
