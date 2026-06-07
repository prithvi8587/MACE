import os
import uuid
import urllib.parse
import requests
from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

app = Flask(__name__)

STATIC_DIR = os.path.join(app.root_path, 'static')
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

import google.generativeai as genai
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct

class EquivariantInteractionBlock(torch.nn.Module):
    def __init__(self, in_irreps, out_irreps, edge_sh_irreps):
        super().__init__()
        self.in_irreps = o3.Irreps(in_irreps)
        self.out_irreps = o3.Irreps(out_irreps)
        self.edge_sh_irreps = o3.Irreps(edge_sh_irreps)

        self.tp = FullyConnectedTensorProduct(
            self.in_irreps,
            self.edge_sh_irreps,
            self.out_irreps
        )
        self.linear = o3.Linear(self.out_irreps, self.out_irreps)

    def forward(self, x, edge_index, edge_vec):
        row, col = edge_index
        edge_sh = o3.spherical_harmonics(self.edge_sh_irreps, edge_vec, normalize=True, normalization='component')

        messages = self.tp(x[col], edge_sh)

        out = torch.zeros(x.size(0), self.out_irreps.dim, device=x.device, dtype=messages.dtype)
        out.index_add_(0, row, messages)
        return self.linear(out)


class ProductionEquivariantMace(nn.Module):
    def __init__(self, node_embed_dim=64, out_tasks=12):
        super().__init__()
        self.emb = nn.Embedding(119, node_embed_dim)

        self.in_irreps = o3.Irreps(f"{node_embed_dim}x0e")
        self.hidden_irreps = o3.Irreps("32x0e + 8x1o + 4x2e")
        self.edge_sh_irreps = o3.Irreps("1x0e + 1x1o + 1x2e")

        self.block1 = EquivariantInteractionBlock(self.in_irreps, self.hidden_irreps, self.edge_sh_irreps)
        self.block2 = EquivariantInteractionBlock(self.hidden_irreps, self.hidden_irreps, self.edge_sh_irreps)

        self.energy_head = o3.Linear(self.hidden_irreps, f"{out_tasks}x0e")

    def forward(self, data, compute_forces=False):
        if compute_forces:
            data.pos.requires_grad_(True)

        h = self.emb(data.x)
        h = torch.relu(self.block1(h, data.edge_index, data.edge_vec))
        h = self.block2(h, data.edge_index, data.edge_vec)

        atomic_properties = self.energy_head(h)
        total_energy = global_mean_pool(atomic_properties, data.batch)

        if compute_forces:
            forces = torch.autograd.grad(
                outputs=total_energy,
                inputs=data.pos,
                grad_outputs=torch.ones_like(total_energy),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            return total_energy, -forces

        return total_energy

device = torch.device("cpu")
MODEL_PATH = 'mace_complete_system.pt'

model = None
scaler = None
target_names = ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv"]

def load_system():
    global model, scaler
    if model is None:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model = ProductionEquivariantMace(node_embed_dim=64, out_tasks=12)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        scaler = checkpoint['scaler']

def get_smiles_from_name(name):
    safe_name = urllib.parse.quote(name)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{safe_name}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
    except Exception:
        return None
    return None

def process_input_smiles(smiles, unique_id, r_max=5.0):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None, "Invalid chemical mapping structure."

    extra_descriptors = {
        "Molecular Weight": f"{Descriptors.MolWt(mol):.2f} g/mol",
        "LogP (Lipophilicity)": f"{Descriptors.MolLogP(mol):.2f}",
        "H-Bond Donors": Descriptors.NumHDonors(mol),
        "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
        "Rotatable Bonds": Descriptors.NumRotatableBonds(mol)
    }

    mol = Chem.AddHs(mol)
    img_filename = f"molecule_{unique_id}.png"
    img_path = os.path.join(STATIC_DIR, img_filename)
    Draw.MolToFile(Chem.RemoveHs(mol), img_path, size=(300, 300))

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) < 0: 
        return None, "Failed to embed 3D geometry matrix topology."

    conf = mol.GetConformer()
    Z, pos = [], []
    xyz_coordinates = ""

    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        x_c = conf.GetAtomPosition(i).x
        y_c = conf.GetAtomPosition(i).y
        z_c = conf.GetAtomPosition(i).z
        Z.append(atom.GetAtomicNum())
        pos.append([x_c, y_c, z_c])
        xyz_coordinates += f"{symbol} {x_c:.4f} {y_c:.4f} {z_c:.4f}\n"

    x = torch.tensor(Z, dtype=torch.long)
    pos = torch.tensor(pos, dtype=torch.float)
    batch = torch.zeros(x.size(0), dtype=torch.long)

    if pos.size(0) > 1:
        dist_matrix = torch.cdist(pos, pos, p=2)
        mask = (dist_matrix <= r_max) & (~torch.eye(pos.size(0), dtype=torch.bool))
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_vec = torch.empty((0, 3), dtype=torch.float)

    graph_data = Data(x=x, pos=pos, edge_index=edge_index, edge_vec=edge_vec, batch=batch)
    num_atoms = mol.GetNumAtoms()
    xyz_data = f"{num_atoms}\nGenerated by MACE App\n" + xyz_coordinates

    return (graph_data, img_filename, extra_descriptors, xyz_data), None

def get_ai_interpretation(smiles, properties):
    data_string = ", ".join([f"{k}: {v:.4f}" for k, v in properties.items()])
    prompt = (f"You are an expert computational chemist analyzing deep learning property predictions. "
              f"The molecule {smiles} has the following predicted quantum chemical properties: {data_string}. "
              f"Provide a brief, professional chemical interpretation of these findings.")
    try:
        model_gemini = genai.GenerativeModel('gemini-2.5-flash')
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception:
        return f"AI Interpretation temporarily unavailable."

@app.route('/', methods=['GET', 'POST'])
def index():
    load_system()
    results, interpretation, error, img_file, extra_descriptors, xyz_data = [None]*6
    smiles_input = ""

    if request.method == 'POST':
        user_input = request.form.get('smiles', '').strip()
        if user_input:
            if not any(char in user_input for char in ['=', '(', ')', '#']) and (' ' in user_input or len(user_input) > 2):
                resolved_smiles = get_smiles_from_name(user_input)
                if resolved_smiles:
                    smiles_input = resolved_smiles
                else:
                    error = f"Could not map compound identity for: '{user_input}'"
            else:
                smiles_input = user_input

            if smiles_input and not error:
                unique_id = uuid.uuid4().hex[:8]
                data_package, err = process_input_smiles(smiles_input, unique_id)
                if data_package is None:
                    error = err
                else:
                    graph_data, img_file, extra_descriptors, xyz_data = data_package
                    with torch.no_grad():
                        preds = model(graph_data)
                    real_preds = scaler.inverse_transform(preds.numpy())[0]
                    results = dict(zip(target_names, real_preds))
                    interpretation = get_ai_interpretation(smiles_input, results)

    return render_template('index.html', results=results, interpretation=interpretation, 
                           error=error, smiles=smiles_input, img_file=img_file, 
                           extra_descriptors=extra_descriptors, xyz_data=xyz_data)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
