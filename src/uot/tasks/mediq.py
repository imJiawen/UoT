import os
import json

from src.uot.chat_utils import import_prompts_by_task
from src.uot.uot import UoTNode

base_dir = os.getcwd()

class MediQTask:
    def __init__(self, args):
        self.__dict__.update(vars(args))
        self.free_answer = True
        self.max_turn = 5
        self.prompts = import_prompts_by_task("mediq")
        self.set = []
        self.data = self.load_dataset(args.dataset)
        self.root = None

    def load_dataset(self, name):
        if name == "icraftmd":
            return json.loads(os.path.join(os.path.dirname(__file__), f"../data/all_craft_md.json").read())
        elif name == "imedqa":
            return json.loads(os.path.join(os.path.dirname(__file__), f"../data/all_dev_good.json").read())
        else:
            raise NotImplementedError
        

    # def create_root(self, root=None):
    #     if not root:
    #         self.root = UoTNode("ROOT", True, self.set, None, self.guesser_model)
    #     else:
    #         root.set_config(self.n_extend_layers, not self.none_acc_reward, self.expected_reward_method)
    #         self.root = root

    def get_omega_for_index(self, idx):
        """
        每个样本自己的 Ω: 这一题的 4 个选项文本列表
        """
        sample = self.data[idx]
        return list(sample["options"].values())

    def create_root(self, items, root=None):
        """
        用当前样本的 Ω 创建 UoT 的根节点
        """
        if root is None:
            self.root = UoTNode("ROOT", True, items, None, self.guesser_model)
        else:
            root.set_config(
                self.n_extend_layers,
                not self.none_acc_reward,
                self.expected_reward_method,
            )
            self.root = root

    def build_patient(self, idx, args):
        """
        Patient
        """
        if self.patient_cls is None:
            raise RuntimeError("patient_cls is not set for MedIQTask")
        sample = self.data[idx]
        return self.patient_cls(args, sample)
