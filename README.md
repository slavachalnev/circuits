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

## Thoughts

---
### layernorm
For one-layer models, I apply layernorm to each embedding vector individually. This doesn't work for two-layer models. Instead, we can roll the layernorm into the weights like this:

Layernorm first subtracts the mean activation. This is equivalent to zeroing-out the direction corresponding to the (1, 1, ..., 1) vector (it's a diagonal line). To get a matrix M such that $Mx$ is equivalent to subtracting the mean from $x$, we can do $M = I - \frac{A}{\text{dim(A)}}$ where A is the matrix of ones.

We then multiply by the learned weights. The division by variance can be factored out. I'm not sure what to do about the bias term.

---
### The <|START|> token
I started out training one-layer models without the <|START|> token. This meant that I had to compute the average qk value for each query and then subtract this from the queries. This seems to work okay, but it's a bit annoying to compute the average qk values. It looks like the <|START|> token is even more useful for two-layer models so I've now started training with it.

---
### dropout
I've found that adding dropout seems to make the positional heads easier to interpret. I suspect this is because the model has to learn to use all of the dimensions of the positional embeddings.
