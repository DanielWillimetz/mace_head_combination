import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ase.io import read
from mace.calculators import MACECalculator
from mace.tools.scripts_utils import remove_pt_head

class Attentionpoolingfraction(nn.Module):
    def __init__(self, *X, pool_type="avg", temperature=0.5):
        super(Attentionpoolingfraction, self).__init__()
        self.X = X
        self.pool_type = pool_type
        self.temperature = temperature
        
    def forward(self, x, block=None):
        query = x
        query = F.normalize(query, p=2, dim=-1)
        pooled = []
        for Xi in self.X:

            if block == "LinearReadoutBlock":
                Xi = Xi[:, 0:1, :].squeeze(1)
            elif block == "NonLinearReadoutBlock":
                Xi = Xi[:, 1:2, :].squeeze(1)
            else:
                raise ValueError("Invalid block")
                
            key = Xi 
            key = F.normalize(key, p=2, dim=-1)
            attn_scores = torch.matmul(query, key.T) / (
                key.size(-1) ** 0.5
            ) 
            if self.pool_type == "avg":
                pooled_score = torch.mean(attn_scores, dim=-1, keepdim=True)
            elif self.pool_type == "max":
                pooled_score, _ = torch.max(attn_scores, dim=-1, keepdim=True)
            else:
                raise ValueError(f"Invalid pool_type: {self.pool_type}. Choose 'avg' or 'max'.")
            
            pooled.append(pooled_score)

        pooled = torch.cat(pooled, dim=-1)
        delta = F.softmax(pooled / self.temperature, dim=-1)

        return delta.T

class ModelWithSimilarity:
    def __init__(self, model_name, X1, X2, pool_type='avg', temperature=0.3):
        self.model_name = model_name
        self.X1 = X1
        self.X2 = X2
        self.pool_type = pool_type
        self.temperature = temperature
        
        self.model = torch.load(model_name)
        
        self.device_calc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = [list(block.children())[0].irreps_in.dim for block in self.model.readouts]

    def generate_tensor(self, database):
        heads = self.model.heads
        for head in heads:
            model_single = remove_pt_head(self.model, head)
            model_single_name = f"single_{head}.model"
            torch.save(model_single, model_single_name)

            calc = MACECalculator(model_paths=[model_single_name], device=str(self.device_calc))
            block_features = torch.empty(0, len(self.input_dim), self.input_dim[0], device=self.device_calc)
            
            atoms_list = read(database, index=":")
            for atoms in atoms_list:
                atoms.calc = calc
                calc.calculate(atoms=atoms)
                features = calc.results["node_feats"]
                blocks_i = torch.stack(torch.split(features, self.input_dim, dim=1), dim=1)
                block_features = torch.cat((block_features, blocks_i), dim=0)

            os.remove(model_single_name)

        return block_features

    def create_similarity_fn(self):
        X1_features = self.generate_tensor(self.X1)
        X2_features = self.generate_tensor(self.X2)
        
        similarity_fn = Attentionpoolingfraction(X1_features, X2_features, pool_type=self.pool_type, temperature=self.temperature)
        
        self.model.similarity_fn = similarity_fn

        new_model_name = f"sim_{self.model_name}"
        torch.save(self.model, new_model_name)

        return self.model
