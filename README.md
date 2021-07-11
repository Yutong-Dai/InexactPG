# Parameter tuning

datasets: 

```
"a9a", "colon_cancer", "duke", "leu", "mushrooms", "w8a"
```

group setup:

```
overlap ratio: [0.1, 0.3, 0.5]
group size: [10, 100]
```

sparsity:

```
0.1, 0.05
```

6 * 2 * 2 * 3 = 72 instances

# Natural Formulation

* Option I:

```
gamma1 = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]
```

* Option II:

```
gamma2 = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]
```

* Option III:

```
c = [1e1, 1e2, 1e3, 1e4, 1e5]
```