# Transformer Circuits
![induction](analysis/assets/induction.png)

Implementation of Anthropic's A Mathematical Framework for Transformer Circuits paper

https://transformer-circuits.pub/2021/framework/index.html

### Attention Visualisation:
https://slavachalnev.github.io/circuits/analysis/potter.html

## Project Layout
```
circuits
|-- analysis                        Reproduce graphs and tables
|   |-- assets                      Images of plots
|   |   |-- one_layer_eigen.png     Eigenvalue plot
|   |   |-- head_11_pos.png         Positional attention head
|   |-- tests                       Test analysis tools
|   |   |-- tests.py                Test weight extractor
|   |-- one_layer.md                One layer results
|   |-- one_layer.py                One layer circuits, \
|   |                               skip-trigrams, eigenvalues
|   |-- potter.html                 Harry Potter analysis
|   |-- two_layer.md                Two layer results
|   |-- two_layer.py                q, k, v composition,
|   |                               attention visualisation
|   |-- utils.py                    Tools to extract head weights
|-- circuits                        Trains attn-only models
|   |-- models                      Define models
|   |   |-- model.py                Base attn-only model
|   |   |-- one_attn_layer.py       One layer transformer
|   |   |-- two_attn_layer.py       Two layer transformer
|   |-- train                       Train and data prep
|   |   |-- openwebtext.py          Prep webtext dataset
|   |   |-- train_one_layer.py      One layer training script
|   |   |-- train_two_layer.py      Two layer training script
|   |   |-- trainer.py              Training loop
|   |   |-- utils.py                Setup logging and seed.
|-- .gitignore                      
|-- LICENCE                         MIT license
|-- README.md
|-- requirements.txt
```

## Notable Differences
- Tokenizer: I use the GPT-2 tokenizer. I don't know what tokenizer the paper used but it's definitely different.
- Dataset: I use the OpenWebText dataset whereas the paper uses a mix of "Common Crawl data and internet books, along with a number of smaller distributions, including about 10% python code data"

## TODO
- Complete attention heads dump (OV/QK values for every token)
- Scond layer eigenvalues
- Term importance analysis
- Publish checkpoints
