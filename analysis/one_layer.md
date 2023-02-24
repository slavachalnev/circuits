
## One Layer Attention-Only Model

### OV Circuit Eigenvalues
![OV Eigenvalues](assets/one_layer_eigen.png)
We see that most of the heads are copying.


### Primarily Positional heads
An example of a head which attends to relative positions.
![Positional head](assets/head_11_pos.png)


```
Source token is " couldn"

head 11
positional max:  0.6577057866183482
positional argmax: 1
# we see that the head is attending to the previous token

source to out
[b'\x99', b'bodied', b' resist', b' contain', b' unwilling']
[0.12821078300476074, 0.12735702097415924, 0.12677621841430664, 0.12531062960624695, 0.1235443577170372]
source to dest
[b't', b' really', b"'t", b' only', b' even']
[22.910142558297608, 18.76203904619527, 18.566329625278822, 17.632106478644054, 17.53535245989735]
```
