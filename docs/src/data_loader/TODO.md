# DATA Loader TODO 

- [x] Implement `@views` instead of allocating a new array in every step. 
- [x] Implement **sampling without replacement**.
- [x] Store information on the epoch and the current loss. 
- [x] Usually the training loss is computed over the entire data set, we are probably going to do this for one epoch via 
```math
loss_e = \frac{1}{|batches|}\sum_{batch\in{}batches}loss(batch).
```

Point 4 makes sense because the output of an AD routine is the value of the loss function as well as the pullback. 