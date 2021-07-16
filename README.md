# DeepMind COVID-19 toolkits

This repository contains tools for working with COVID-19 data and forecasting.
It contains:

*   A [Python-based evaluation toolkit] for standardising the data format and
    comparison of COVID-19 forecasts.
*   A [starter codebase for creating machine-learning-based forecasting models],
    with reference implementations and training scripts, compatible with the
    evaluation toolkit. This codebase is built with Python and the reference
    implementations are written in [JAX].

Please see the links above for documentation and installation instructions.

If you use our codebase, please cite our work as:

```
@software{deepmind2021c19modelling,
  author = {Sanchez-Gonzalez, Alvaro and Mottram, Anne and Doron, Yotam and Hapuarachchi, Tharindi and Horstmann, Tobias and Hou, Shaobo and Hughes, Cian  and Irimia, Valentin and Karthikesalingam, Alan and Landon, Jessica and Mokr\'{a}, So\v{n}a and Petrova, Dessie and Rosen, Simon and Toma\v{s}ev, Nenad and Vinyals, Oriol and Ward, Michael and Battaglia, Peter W},
  title = {DeepMind Covid-19 evaluation and modelling framework.},
  url = {http://github.com/deepmind/dm_c19_modelling},
  version = {0.0.1.dev},
  year = {2021},
}
```

(A.S-G., A.M. and P.B. share equal contribution, rest of authors in alphabetical order).


[JAX]: https://github.com/google/jax
[Python-based evaluation toolkit]: evaluation/README.md
[starter codebase for creating machine-learning-based forecasting models]: modelling/README.md
