import argparse
from Trainer.methods import TemporalPINN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--device', type=str, default="cpu", help="device: cpu or cuda:id")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--iters', type=int, default=1)

    parser.add_argument('--n_int', type=int, default=10000)
    parser.add_argument('--n_sb', type=int, default=100)
    parser.add_argument('--nt', type=int, default=10)
    parser.add_argument('--delta_t', type=float, default=0.01)
    parser.add_argument('--save_path', type=str, default='./results/ADAM_test_1.pt')
    parser.add_argument('--pre_model_save_path', type=str, default=None)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--lr', type=float, default=1e-3)


    args=parser.parse_args()
    assert args.train or args.test, "Please specify --train or --test"

    if args.train:
        TemporalPINN(args.n_int, args.n_sb, args.nt, args.delta_t, args.epochs, args.device, args.seed, args.save_path, args.pre_model_save_path, args.optimizer, args.lr, args.iters)

