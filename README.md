# The Nelder-Mead method

An alternative implementation of the Nelder-Mead method, there is no need of knowing the function being optimized. The code is completely implemented in python.

## Description

This implementation is based on the following article:

###[http://www.scholarpedia.org/article/Nelder-Mead_algorithm#Initial_simplex](http://www.scholarpedia.org/article/Nelder-Mead_algorithm#Initial_simplex)

<!-- ![result](https://github.com/owruby/nelder_mead/blob/master/figures/anim.gif) -->

### Dependencies

- numpy

If you wish to run the examples the **matplotlib** dependency will also be needed.

## Usage

### Example 1:

``` python
from nelder_mead import NelderMead

def sphere(x):
    return sum([value**2 for value in x])

def main():
    f_variables = np.array([1.0, 1.0])
    nelder = NelderMead(f_variables)

    for step in range(0, 30):
        f_value = sphere(f_variables)
        nelder.run(f_value)

if __name__ == "__main__":
    main()
```

The NelderMead class also allow you to change the values of the transformation parameters:

## References

- http://www.scholarpedia.org/article/Nelder-Mead_algorithm#Initial_simplex
- https://codesachin.wordpress.com/2016/01/16/nelder-mead-optimization/
- https://stackoverflow.com/questions/17928010/choosing-the-initial-simplex-in-the-nelder-mead-optimization-algorithm
