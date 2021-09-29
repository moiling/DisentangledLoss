### Disentangled Loss for Trimap Generation.

We regard the Trimap Generation problem as two steps. <br /> 
The first step is to distinguish the foreground and background, <br />
and the second step is to distinguish the opaque region and transparent region in the foreground. <br />
In this way, the Segmentation dataset and the Matting dataset can be used for training respectively.

```
disentangled_loss = background_loss(bu_loss) + transparent_loss(fu_loss) 

background_loss  = - 1/N \Sum_i^N [M_B\log(p_B)+M_U\log(p_F+p_U)]
transparent_loss = - 1/N \Sum_i^N [M_F\log(p_F)+M_U\log(p_B+p_U)]

M_x means the mask of region x labeled in trimap.
p_x means the probability of region x (after softmax).
```

```
                                                        p_B ----------->
SegDataset -> seg_img -> network -> pred_trimap_prob -> p_U ->         | <- loss -> seg_label
                            |                                + -> p_U ->
                            |                           p_F ->
                          same
                            |                           p_B ->
                            |                                + -> p_U ->
MatDataset -> mat_img -> network -> pred_trimap_prob -> p_U ->         | <- loss -> trimap_label (union B & U)
                                                        p_F ----------->

```