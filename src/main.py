from timeit import default_timer

from transfer_learning import save_vgg_features, train_top_layers


def main():
    save_vgg_features()
    train_top_layers()


if __name__ == '__main__':
    start = default_timer()
    main()
    duration = (default_timer() - start) / 60
    print('The process took {0:2f} minutes'.format(duration))
