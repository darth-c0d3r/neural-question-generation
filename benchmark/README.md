# Benchmark Results

## 1. Effect of Fast Tokenizer

Tokenizer: `sshleifer/distilbart-cnn-6-6` <br>
Device: `Colab CPU` (doesn't matter in this case)

| Batch Size / Seq. Len. |        128       |        256       |        512       |       1024       | Average Speedup |
| ---------------------- | ---------------- | ---------------- | ---------------- | ---------------- | --------------- |
|            8           | 0.5053 -> 0.0047 | 0.4692 -> 0.0073 | 0.9647 -> 0.0139 | 1.2278 -> 0.0279 |      71.3x      |
|           16           | 0.3364 -> 0.0078 | 0.7021 -> 0.0143 | 1.3868 -> 0.0288 | 2.5529 -> 0.0518 |      47.4x      |
|           32           | 0.6308 -> 0.0144 | 1.2394 -> 0.0270 | 2.4772 -> 0.0528 | 4.9720 -> 0.1031 |      46.2x      |
|           64           | 1.2655 -> 0.0293 | 2.5135 -> 0.0536 | 5.2116 -> 0.1080 | 9.8045 -> 0.2034 |      46.6x      | 
|     Average Speedup    |      59.4x       |      51.5x       |      53.2x       |      47.4x       |      52.9x      |

`->` indicates the change in inference time from normal to fast tokenizer.  <br>
All numbers are reported for the average time taken (in seconds) to finish a single batch.

**Conclusion** : Always use Fast Tokenizers if available.

## 2. Model Benchmarks on GPU

Model: `sshleifer/distilbart-cnn-6-6` <br>
Device: `Tesla K80`

**Beam Size = 1 [Greedy Decoding]**

| Batch Size / Seq. Len. |   128  |   256  |   512  |
| ---------------------- | ------ | ------ | ------ |
|            8           | 1.1371 | 1.6032 | 1.8484 |
|           16           | 1.7534 | 2.1401 | 2.7694 |
|           32           | 2.3992 | 2.8456 | 4.3321 |


**Beam Size = 2**

| Batch Size / Seq. Len. |   128  |   256  |   512  |
| ---------------------- | ------ | ------ | ------ |
|            8           | 2.0029 | 2.1392 | 2.7203 | 
|           16           | 2.8036 | 3.2537 | 4.0959 | 
|           32           | 4.4048 | 5.4221 | 7.8130 | 


**Beam Size = 3**

| Batch Size / Seq. Len. |   128  |   256  |   512  |
| ---------------------- | ------ | ------ | ------ |
|            8           | 2.5252 | 2.2454 | 3.2476 | 
|           16           | 3.5154 | 4.0599 | 5.0957 | 
|           32           | 6.0350 | 7.6085 | 10.079 | 


**Beam Size = 4**

| Batch Size / Seq. Len. |   128  |   256  |   512  |
| ---------------------- | ------ | ------ | ------ |
|            8           | 2.7710 | 2.7367 | 3.7545 | 
|           16           | 3.9262 | 4.8438 | 6.0295 | 
|           32           | 7.5984 | 9.0678 | 12.080 | 

All numbers are reported for the average time taken (in seconds) to finish a single batch. <br>
Use these raw numbers as reference for comparison with other benchmarks.

## 3. Effect of Quantization

Model: `sshleifer/distilbart-cnn-6-6` <br>
Device: `CPU` (Quantization only supports CPU)

**Beam Size = 1 [Greedy Decoding]**

| Batch Size / Seq. Len. |            256            |            512            | Average Speedup |
| ---------------------- | ------------------------- | ------------------------- | --------------- |
|           16           | 26.7303 -> 16.7891 [1.6x] | 41.4051 -> 27.5787 [1.5x] |       1.55x     |
|           32           | 45.1856 -> 30.8891 [1.5x] | 72.6482 -> 55.2227 [1.3x] |       1.40x     |
|     Average Speedup    |           1.55x           |           1.40x           |       1.47x     |


**Beam Size = 4**

| Batch Size / Seq. Len. |            256            |            512            | Average Speedup |
| ---------------------- | ------------------------- | ------------------------- | --------------- |
|           16           | 58.3237 -> 40.7531 [1.4x] | 82.7789 -> 63.3424 [1.3x] |       1.35x     |
|           32           | 116.753 -> 80.3119 [1.5x] | 165.465 -> 124.361 [1.3x] |       1.40x     |
|     Average Speedup    |           1.45x           |           1.30x           |       1.37x     |


All numbers are reported for the average time taken (in seconds) to finish a single batch. <br>

**Conclusions** : 

* You can expect a speedup of ~1.4x if you use dynamic quantization on your model directly on CPU.
* However the eval loss goes from `1.499` to `3.828` and the resulting model outputs are visibly worse.
* Therefore, without quantization aware training, this is not a viable optimization for text generation.
* Also, using a GPU is just much faster as compared to CPU (more than 12x speedup).
