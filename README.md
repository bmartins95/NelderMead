# The Nelder-Mead method

An alternative implementation of the Nelder-Mead method, there is no need of knowing the function being optimized. The code is completely implemented in python.

## Description

This implementation is based on the following article:

### [http://www.scholarpedia.org/article/Nelder-Mead_algorithm#Initial_simplex](http://www.scholarpedia.org/article/Nelder-Mead_algorithm#Initial_simplex)

<p align="center">
    <img src="https://github.com/bmartins95/NelderMead/blob/master/figures/paraboloid.gif">
</p>

### Installation

```
pip install git+https://github.com/bmartins95/NelderMead
```

### Dependencies

- numpy

If you wish to run the examples the **matplotlib** dependency will also be needed.

## Usage

### Example 1:

The most simple example of the class usage is:

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

### Example 2:

You can change the initial simplex construction by inheriting the NelderMead class and modifying the *buildSimplexPoints* function as follows:

``` python
    class NewNelderMead(NelderMead):
        def buildSimplexPoints(self):
            x0 = np.array([1.0, 1.0])
            x1 = np.array([2.5, 1.0])
            self.simplex = np.vstack((x0, x1, self.f_variables))
```

The last line of the simplex must aways be **f_variables**.

## References

- http://www.scholarpedia.org/article/Nelder-Mead_algorithm#Initial_simplex
- https://codesachin.wordpress.com/2016/01/16/nelder-mead-optimization/
- https://stackoverflow.com/questions/17928010/choosing-the-initial-simplex-in-the-nelder-mead-optimization-algorithm
