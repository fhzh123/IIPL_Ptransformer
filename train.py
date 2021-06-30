import trainer
import argparse

def main(param):
    model = trainer.Trainer(file_path=param.file, token_type=param.token_type, load=param.load,
                            emb_size=param.emb_size, num_epoch=param.num_epoch, nhead=param.nhead, 
                            ffn_hid_dim=param.ffn_hid_dim, batch_size=param.batch_size,
                            n_layers=param.n_layers, dropout=param.dropout, variation=param.variation)
    model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Machine Translation')
    parser.add_argument('--token_type', type=int, default=1, choices=[1, 2])
    parser.add_argument('--file', type=str, default="Default")
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--num_epoch', type=int, default=15)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--emb_size', type=int, default=512)
    parser.add_argument('--ffn_hid_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--variation', type=bool, default=False)
    args = parser.parse_args()
    main(args)