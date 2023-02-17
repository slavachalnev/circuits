# circuits
Implementation of Anthropic's transformer circuits paper

https://transformer-circuits.pub/2021/framework/index.html

## Project Layout
```
circuits
|-- analysis                        Reproduce graphs and tables
|   |-- assets                      Images of plots
|   |   |-- one_layer_eigen.png     Eigenvalue plot
|   |   |-- head_5_pos.png          Positional attention head
|   |-- tests                       Test analysis tools
|   |   |-- tests.py                Test weight extractor
|   |-- one_layer.md                One layer results
|   |-- one_layer.py                One layer circuits, \
|   |                               skip-trigrams, eigenvalues
|   |-- zero_layer_chars.py         Learn bigram char stats
|-- circuits                        Trains attn-only models
|   |-- models                      Define models
|   |   |-- model.py                Base attn-only model
|   |   |-- one_attn_layer.py       One layer transformer
|   |   |-- two_attn_layer.py       Two layer transformer
|   |   |-- zero_layer.py           Bigram model
|   |-- train                       Train and data prep
|   |   |-- openwebtext.py          Prep webtext dataset
|   |   |-- train_one_layer.py      One layer training script
|   |   |-- train_two_layer.py      Two layer training script
|   |   |-- trainer.py              Training loop
|   |   |-- utils.py                Setup logging and seed.
|-- .gitignore                      
|-- LICENCE                         MIT license
|-- README.md

```

## What I'm thinking about

---
### layernorm
I currently roll the first layernorm into the embedding matrix by applying layernorm to each embedding vector individually. This won't work for two-layer models.
It also doesn't work for the final layernorm (which I'm pretending doesn't exist even though I train with it 👀).

Layernorm first subtracts the mean activation. This is equivalent to zeroing-out the direction corresponding to the (1, 1, ..., 1) vector (it's a diagonal line). I can think of two ways of finding a matrix M that does this:

1. We can find a rotation matrix R which would rotate the x-axis to lie on the diagonal. It is a simple rotation in the plane spanned by the x-axis and the diagonal. $M = R^TAR$ and A is a matrix that squashes the x-axis.

2. Alternatively, we can find an orthonormal basis for $\mathbb{R}^n$ which includes the diagonal as one of the basis vectors. We can use the Gram-Schmidt process to find an orthonormal basis. $M = VAV^{-1}$ where V is the matrix whose columns are the basis vectors and A is a diagonal matrix that squashes the x-axis.

I will take the second approach. TODO: implement this.

We then multiply by the learned weights. The division by variance can be factored out. I'm not what to do about the bias term.

---
### The <|START|> token
I started out training one-layer models without the <|START|> token. This meant that I had to compute the average qk value for each query and then subtract this from the queries. This seems to work okay, but it's a bit annoying to compute the average qk values. It looks like the <|START|> token is even more useful for two-layer models so I've now started training with it.

