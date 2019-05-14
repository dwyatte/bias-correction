cat mnist.sh cifar10.sh cifar100.sh | parallel -j 4 'export CUDA_VISIBLE_DEVICES=$(({%}-1)) && eval {}'
