```mkdir data```

To run active queries

```./place_flowers_softmax [seed] 0```

```./plot_spoon_softmax [seed] 0```

To run random queries

```./place_flowers_softmax [seed] 1```

```./plot_spoon_softmax [seed] 1```



To create visualizations of learned reward and VaR reward queries

```./place_flowers_softmax_notest [seed] 0```

or 

```./place_spoon_softmax_notest [seed] 0```

then run 
```python plot_flower_query_frames.py```
