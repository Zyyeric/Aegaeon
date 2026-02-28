# Model Swapping Simulator 

This is a simulator for understanding how the model swapping affects scheduling/placement policies and the end-to-end serving performance. The implementation is based on [SimPy](https://simpy.readthedocs.io/en/latest).

### Estimations

To accurately simulate the performance of any scheduling/placement policy, it is essential to estimate the following with precision:

#### Model execution time

For LLMs, this includes both the prefill and the decode phase, with respect to the batch size, number of input tokens, parallelism settings and other related parameters.

We follow the practice proposed in [DistServe](https://arxiv.org/pdf/2401.09670) (Appendix A) to fit a latency model for LLM prefill and decode phases. 

In short, the latency of the prefill phase can be modeled as:

$$T_{\text{prefill}} = C_1\cdot (4th^2+2thm) + C_2 \cdot \frac{3ht_2}{b} + C_3$$

While the latency of the decoding phase (one iteration) is:

$$T_{\text{decoding}} = C_4\cdot (4h^2+2hm) + C_5 \cdot 3ht$$

where the definitions of symbols are given below:

* $h$: hidden size
* $m$: FFN intermediate size
* $t$: number of tokens in the batch (some of all sequence lengths)
* $t_2$: squared sum of the input lengths
* $b$: block size in the attention kernel (used in FlashAttention, which is the default backend in vLLM)

We then fit the latency model against profiling results to figure out the values of coefficients.


#### Model swapping time 
In general, this covers all the overhead involved between having two running LLM instances. Ideally this should reduce to only the model loading time (i.e. time it takes to transfer weights through PCIe/NVLINK to the target GPU) and a negligible amount of other overhead, such as connecting to existing ray clusters and garbage collecting Python objects.

### Traces

We take several real traces and merge them to produce verbose request traces:

* **Arrival time**. Since the trace only contains QPS (query per second) at the granularity of one minute, we treat each minute as a Poisson process with varying rates ($\lambda$), and reconstruct the arrival times by accumulating exponential interval times (with a mean of $1/\lambda$).

* **Input and output lengths**. We randomly sample pairs of input and output lengths from another set of traces and assign them to the reconstructed requests.