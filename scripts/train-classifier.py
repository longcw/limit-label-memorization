from methods import classifiers
from modules import training
import modules.data as datasets
import modules.visualization as vis
import argparse
import importlib
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--device', '-d', default='cuda')
    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--epochs', '-e', type=int, default=400)
    parser.add_argument('--save_iter', '-s', type=int, default=10)
    parser.add_argument('--vis_iter', '-v', type=int, default=2)
    parser.add_argument('--log_dir', '-l', type=str, default=None)
    parser.add_argument('--dataset', '-D', type=str, default='mnist',
                        choices=['mnist'])
    parser.add_argument('--noise_level', '-n', type=float, default=0.0)
    parser.add_argument('--encoder_path', '-r', type=str, default=None)
    parser.add_argument('--model_class', '-m', type=str, default='StandardClassifier',
                        choices=['StandardClassifier', 'PretrainedClassifier',
                                 'PenalizeLastLayerFixedForm', 'PenalizeLastLayerGeneralForm',
                                 'PredictGradOutputFixedForm', 'PredictGradOutputGeneralForm',
                                 'PredictGradOutputMetaLearning'])
    parser.add_argument('--train_encoder', dest='freeze_encoder', action='store_false')
    parser.add_argument('--grad_weight_decay', '-L', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lamb', type=float, default=0.0)
    parser.add_argument('--num_train_examples', type=int, default=None)
    parser.add_argument('--nsteps', type=int, default=1)
    parser.set_defaults(freeze_encoder=True)
    args = parser.parse_args()
    print(args)

    # Load data
    if args.dataset == 'mnist':
        train_loader, val_loader, test_loader = datasets.load_mnist_loaders(
            batch_size=args.batch_size, noise_level=args.noise_level,
            num_train_examples=args.num_train_examples)

    example_shape = train_loader.dataset[0][0].shape
    print("Dataset is loaded:\n\ttrain_samples: {}\n\tval_samples: {}\n\t"
          "test_samples: {}\n\tsample_shape: {}".format(
        len(train_loader.dataset), len(val_loader.dataset),
        len(test_loader.dataset), example_shape))

    # Options
    optimization_args = {
        'optimizer': {
            'name': 'adam',
            'lr': 1e-3
        }
    }

    with open(args.config, 'r') as f:
        architecture_args = json.load(f)

    model_class = getattr(classifiers, args.model_class)

    model = model_class(input_shape=train_loader.dataset[0][0].shape,
                        architecture_args=architecture_args,
                        encoder_path=args.encoder_path,
                        device=args.device,
                        freeze_encoder=args.freeze_encoder,
                        grad_weight_decay=args.grad_weight_decay,
                        weight_decay=args.weight_decay,
                        lamb=args.lamb,
                        nsteps=args.nsteps)

    training.train(model=model,
                   train_loader=train_loader,
                   val_loader=val_loader,
                   epochs=args.epochs,
                   save_iter=args.save_iter,
                   vis_iter=args.vis_iter,
                   optimization_args=optimization_args,
                   log_dir=args.log_dir)

    # do final visualizations
    if hasattr(model, 'visualize'):
        visualizations = model.visualize(train_loader, val_loader)

        for name, fig in visualizations.items():
            vis.savefig(fig, os.path.join(args.log_dir, name, 'final.png'))


if __name__ == '__main__':
    main()
