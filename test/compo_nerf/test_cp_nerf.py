import unittest
from src.compo_nerf.cp_nerf import CompoNeRF
from src.compo_nerf.configs.train_config import TrainConfig
import pyrallis

class TestCPNeRF(unittest.TestCase):
    def __init__(self, methodName: str = "runTest", ) -> None:
        super().__init__(methodName)
        self.cfg = pyrallis.parse(config_class=TrainConfig, config_path='abls_configs/table_wine_box_learn.yaml')
        # python -m test.compo_nerf.test_cp_nerf.TestCompoNeRF.test_init_nodes