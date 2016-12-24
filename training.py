from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

from model import train_dgcan


def main():

    class Config(object):
        pass

    config = Config()
    config.epochs = 10
    config.batch_size = 32
    config.learning_rate = 0.0002
    config.beta1 = .5
    config.image_size = 28
    config.image_channels = 1

    dataset = MNIST(which_sets=('train',),
                    sources=('features', 'targets'))
    stream = DataStream.default_stream(
        dataset=dataset,
        iteration_scheme=ShuffledScheme(
            examples=dataset.num_examples,
            batch_size=config.batch_size))

    train_dgcan(stream, config)


if __name__ == '__main__':
    main()
