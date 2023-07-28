import pyrallis

from src.compo_nerf.configs.train_config import TrainConfig
from src.compo_nerf.training.trainer import Trainer
import os
os.environ['CURL_CA_BUNDLE'] = ''

@pyrallis.wrap()
def main(cfg: TrainConfig):
    trainer = Trainer(cfg)
    if cfg.log.eval_only:
        trainer.full_eval()
    else:
        trainer.train()

if __name__ == '__main__':
    main()