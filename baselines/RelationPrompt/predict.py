import json
import random
from pathlib import Path
from wrapper import Generator, Extractor, Dataset
import argparse


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--seed", default=2, type=int, required=True, help="random seed")
    parser.add_argument("--ratio", default="0.01", type=str, required=True, help="split ratio")
    parser.add_argument("--data_name", default="webnlg_retrieved", type=str, required=True, help="data name")
    parser.add_argument("-lr", "--learning_rate", default=5e-4, type=float, required=True, help="learning_rate")
    parser.add_argument("--eval_steps", default=20, type=int, required=True, help="eval steps")

    args = parser.parse_args()
    seed = args.seed
    ratio = args.ratio
    data_name = args.data_name
    lr = args.learning_rate
    base_data = data_name.split("_")[0]
    epoch = 35 if base_data == "nyt" else 45

    data_dir = f"outputs/data/{data_name}_ratio/seed{seed}/{ratio}"
    path_test = f"{data_dir}/test.json"
    print(dict(data_dir=data_dir))
    save_dir = f"outputs/wrapper/{data_name}_ratio/seed{seed}/{ratio}_lr{lr}_epoch{epoch}"
    print(dict(save_dir=save_dir))
    model_kwargs = dict(batch_size=32, grad_accumulation=4, lr_finetune=float(lr), epochs_finetune=epoch,
                        base_data=base_data)  # For lower memory on Colab

    extractor = Extractor(
        load_dir=str(Path(save_dir) / "extractor" / "model"),
        save_dir=str(Path(save_dir) / "extractor"),
        model_kwargs=model_kwargs,
    )

    path_pred = str(Path(save_dir) / "pred.jsonl")

    extractor.predict_multi(path_in=path_test, path_out=path_pred)
    results = extractor.score(path_pred, path_test)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
