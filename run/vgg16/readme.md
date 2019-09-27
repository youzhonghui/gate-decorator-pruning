This demo show how to use the code from command line:

1. navigate to the `gate-decorator-pruning` folder.
2. `CUDA_VISIBLE_DEVICES=0 python main.py --config ./run/vgg16/baseline.json`
3. `CUDA_VISIBLE_DEVICES=0 python ./run/vgg16/vgg16_prune_demo.py --config ./run/vgg16/prune.json`

The intermediate results can be viewed at the `output.txt`