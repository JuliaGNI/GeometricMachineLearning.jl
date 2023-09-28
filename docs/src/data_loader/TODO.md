# DATA Loader TODO 

1. Implement `@views` instead of allocating a new array in every step. 
2. Implement **sampling without replacement**.
3. Store information on the epoch and the current loss. 
4. Usually the training loss is computed over the entire data set, we are probably going to do this for one epoch via 
```math
loss_e = \frac{1}{|batches|}\sum_{batch\in{}batches}loss(batch).
```

Point 4 makes sense because the output of an AD routine is the value of the loss function as well as the pullback. 