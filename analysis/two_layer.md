## Two-Layer Attention-Only Model

### Q, K and V Composition

![QKV Composition](assets/induction_heads.png)
We see from the array on the right that there are 3 layer-0 previous token heads (heads 10, 11, and 5 in decreasing order of importance). Layer-1 heads 3 and 9 both k-compose with layer-0 head 11.
