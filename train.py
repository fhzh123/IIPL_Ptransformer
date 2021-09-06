import trainer
import argparse

def main(param):
    model = trainer.Trainer(load=param.load, emb_size=param.emb_size, num_epoch=param.num_epoch, nhead=param.nhead, 
                            ffn_hid_dim=param.ffn_hid_dim, batch_size=param.batch_size, lr=param.learning_rate,
                            n_layers=param.n_layers, dropout=param.dropout, variation=param.variation, isP=param.isP)
    model.learn()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Machine Translation')
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--num_epoch', type=int, default=30)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--emb_size', type=int, default=512)
    parser.add_argument('--ffn_hid_dim', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--isP', type=bool, default=False)
    parser.add_argument('--variation', type=str, default="None", choices=["None", "puzzle"])
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    args = parser.parse_args()
    main(args)