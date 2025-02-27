import torch
import numpy as np
from rdkit import Chem
import rdkit.Chem.QED
from coati.common.util import batch_indexable
from coati.utils.sascorer import compute_sa_score
from coati.generative.coati_purifications import force_decode_valid_batch, purify_vector, embed_smiles
import pickle
from tqdm import tqdm
def bump_potential(V, bumps=[], radius=0.125, height=80.0, vec_dim=256):
    """
    Explore space by using gaussian bump potentials when the vector isn't
    changing a lot.
    """

    if len(bumps) < 1:
        return torch.zeros(1, device=V.device)

    bump_potential = (
        height
        * (
            (
                torch.distributions.multivariate_normal.MultivariateNormal(
                    torch.stack(bumps, 0).to(V.device),
                    radius * torch.eye(vec_dim).to(V.device),
                ).log_prob(V)
                + (vec_dim / 2) * (np.log(2 * torch.pi) + np.log(radius))
            ).exp()
        ).sum()
    )

    return bump_potential

def coati_metadynamics(
    init_emb_vec,
    objective_fcn,
    encoder,
    tokenizer,
    constraint_functions=[],  # enforced to == 0 by lagrange multipliers.
    log_functions=[],
    bump_radius=0.125 * 4,
    bump_height=80.0 * 16,
    nsteps=4000,
    save_traj_history=None,
):
    """
    Minimize an objective function in coati space.
    Purifies the vector as it goes along.
    The purified vector satisfies:
      vec \approx purify_vector(vec)

    contraint_functions: list of dict
        routines returning 'constraint_name' : tensor pairs.
    log_functions: list of dict
        routines returning 'value_name' : tensor pairs.

    Returns:
        history: (list of dict). Trajectory history.
    """
    vec = torch.nn.Parameter(init_emb_vec.to(encoder.device))
    vec.requires_grad = True

    # setup optimizer (SGD, Adam, etc.)
    params = [vec]
    for _ in constraint_functions.keys():
        params.append(torch.nn.Parameter(100 * torch.ones_like(vec[:1])))

    # optimizer = torch.optim.Adam(params,lr = 2e-3)
    optimizer = torch.optim.SGD(params, lr=2e-3)

    smiles = force_decode_valid_batch(init_emb_vec, encoder, tokenizer)
    history = [
        {
            "emb": vec.flatten().detach().cpu().numpy(),
            "name": 0,
            "smiles": smiles,
            "library": "opt",
            "activity": objective_fcn(vec).item(),
            **{
                c: constraint_functions[c](vec).detach().cpu().numpy()
                for c in constraint_functions
            },
            **{c: log_functions[c](vec).detach().cpu().numpy() for c in log_functions},
        }
    ]

    # no bumps are initialized.
    bumps = []
    last_bump = 0
    save_every = 25
    project_every = 15
    for k in range(nsteps):
        if k % project_every == 0 and k > 0:
            vec.data = 0.4 * vec.data + 0.6 * purify_vector(
                vec.data, encoder, tokenizer, n_rep=50
            )

        optimizer.zero_grad()
        activity = objective_fcn(vec)

        constraint_values = []
        for f in constraint_functions.keys():
            constraint_values.append(constraint_functions[f](vec))
        if len(constraint_values):
            constraint_term = torch.sum(torch.concat(constraint_values))
        else:
            constraint_term = torch.zeros_like(activity)

        # add a bump_term to the loss (=0 if no bumps).
        bump_term = bump_potential(vec, bumps, radius=bump_radius, height=bump_height)

        loss = activity + constraint_term + bump_term
        loss.backward()  # retain_graph=True)

        if k % save_every == 0:
            # Try to decode the vector here into a molecule!.
            smiles = force_decode_valid_batch(vec, encoder, tokenizer)

            history.append(
                {
                    "emb": vec.flatten().detach().cpu().numpy(),
                    "name": k,
                    "smiles": smiles,
                    "library": "opt",
                    "loss": loss.detach().cpu().item(),
                    "activity": activity.detach().cpu().item(),
                    "bump_term": bump_term.detach().cpu().item(),
                    "const_term": constraint_term.detach().cpu().item(),
                    **{
                        c: log_functions[c](vec).detach().cpu().item()
                        for c in log_functions
                    },
                }
            )

            v1 = history[-1]["emb"]
            v2 = history[-2]["emb"]
            s1 = history[-1]["smiles"]
            s2 = history[-2]["smiles"]

            # build log string
            log_str = f"{k}: dV {np.linalg.norm(v1-v2):.3e} "
            to_log = ["loss", "activity", "bump_term", "const_term"] + list(
                log_functions.keys()
            )
            for l in to_log:
                log_str = log_str + f"{l}:{history[-1][l]:.2f} "

            if (
                ((v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2)) > 0.85)
                and (k - last_bump > 25)
            ) or (s1 == s2 and k > 50):
                print("adding bump ", smiles)
                last_bump = k
                new_bump = torch.from_numpy(v1).to(device=vec.device)
                bumps.append(new_bump)

            if save_traj_history is not None:
                # save trajectory to file
                with open(save_traj_history, "wb") as f:
                    pickle.dump(history, f)

            print(log_str)

        optimizer.step()
    return history

def embed_and_score_in_batches_regression(
    records,
    encoder,
    tokenizer,
    batch_size=128,
    score=True,
    smiles_field="smiles",
):
    """
    records: iterable of dict
    """

    print("Embedding and scoring iterable from smiles.")
    batch_iter = batch_indexable(records, batch_size)
    num_batches = len(records) // batch_size
    with torch.no_grad():
        for i, batch in enumerate(tqdm(batch_iter, total=num_batches, unit="batch")):
            # print(f"batch: {i}/{num_batches}")
            try:
                batch_mols = [Chem.MolFromSmiles(row[smiles_field]) for row in batch]
                batch_smiles = [Chem.MolToSmiles(m) for m in batch_mols]
                batch_tokens = torch.tensor(
                    [
                        tokenizer.tokenize_text("[SMILES]" + s + "[STOP]", pad=True)
                        if s != "*"
                        else tokenizer.tokenize_text("[SMILES]C[STOP]", pad=True)
                        for s in batch_smiles
                    ],
                    device=encoder.device,
                    dtype=torch.int,
                )
                batch_embeds = encoder.encode_tokens(batch_tokens, tokenizer)
                if score:
                    batch_logp = [rdkit.Chem.Crippen.MolLogP(m) for m in batch_mols]
                    batch_qed = [rdkit.Chem.QED.qed(m) for m in batch_mols]
                    batch_sa = [compute_sa_score(m) for m in batch_mols]
                if len(batch) < 2:
                    batch[0]["emb_smiles"] = batch_embeds[0].detach().cpu().numpy()
                    if score:
                        batch[0]["qed"] = batch_qed[0]
                        batch[0]["logp"] = batch_logp[0]
                        batch[0]["smiles"] = batch_smiles[0]
                        batch[0]['sa'] = batch_sa[0]
                else:
                    for k, r in enumerate(batch):
                        batch[k]["emb_smiles"] = batch_embeds[k].detach().cpu().numpy()
                        if score:
                            batch[k]["qed"] = batch_qed[k]
                            batch[k]["logp"] = batch_logp[k]
                            batch[k]["smiles"] = batch_smiles[k]
                            batch[k]['sa'] = batch_sa[k]
            except Exception as e:
                print(e)
                continue