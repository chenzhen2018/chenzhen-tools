[Gradient-based learning applied to document recognition](<https://ieeexplore.ieee.org/document/726791>)

![](https://res.cloudinary.com/chenzhen/image/upload/v1558159479/github_image/2019-05-18/05_18_001.jpg)

|       | input                  | kernel                | output                 | parameter                 | connections                                |
| ----- | ---------------------- | --------------------- | ---------------------- | ------------------------- | ------------------------------------------ |
| Input | $32 \times 32$         |                       |                        |                           |                                            |
| C1    | $32 \times 32$         | $5\times 5\times 6$   | $28\times 28\times 6$  | $(5\times 5 + 1)\times 6$ | $(5 \times 5+1)\times 6\times 28\times 28$ |
| P2    | $28\times 28\times 6$  | $2\times 2\times 6$   | $14\times 14\times 6$  |                           | $(2 \times 2+1)\times 6\times 14\times 14$ |
| C3    | $14\times 14\times 6$  | $5\times 5 \times 16$ | $10\times 10\times 16$ |                           |                                            |
| P4    | $10\times 10\times 16$ | $2\times 2\times 16$  | $5\times 5\times 16$   |                           |                                            |
|       | $5\times 5\times 16$   |                       | $1\times 400 $         |                           |                                            |
| F5    | $1\times 400 $         |                       | $1\times 120 $         |                           |                                            |
| F6    | $1\times 120$          |                       | $1\times 84$           |                           |                                            |
| F7    | $1\times 84$           |                       | $1\times 10$           |                           |                                            |

