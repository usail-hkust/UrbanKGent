import json, sys
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
    eval_steps = args.eval_steps
    save_steps = eval_steps

    base_data = data_name.split("_")[0]
    epoch = 35 if base_data == "nyt" else 45

    data_dir = f"outputs/data/{data_name}_ratio/seed{seed}/{ratio}"
    path_train = f"{data_dir}/train.json"
    path_dev = f"{data_dir}/dev.json"
    path_test = f"{data_dir}/test.json"

    print(dict(data_dir=data_dir))
    save_dir = f"outputs/wrapper/{data_name}_ratio/seed{seed}/{ratio}_lr{lr}_epoch{epoch}"
    print(dict(save_dir=save_dir))
    model_kwargs = dict(batch_size=32, grad_accumulation=4, lr_finetune=float(lr), epochs_finetune=epoch,
                        save_steps=save_steps, eval_steps=eval_steps, base_data=base_data)

    extractor = Extractor(
        load_dir="facebook/bart-base",
        save_dir=str(Path(save_dir) / "extractor"),
        model_kwargs=model_kwargs,
    )

    extractor.fit(path_train, path_dev)
    path_pred = str(Path(save_dir) / "pred.json")

    extractor.predict_multi(path_in=path_test, path_out=path_pred, base_data=base_data)
    results = extractor.score(path_pred, path_test)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
