import sys
from base_path import HUMAN_DATA_PATH

sys.path.append(HUMAN_DATA_PATH + "human_action/human_body_prior")
from human_body_prior.body_model.body_model import BodyModel
import torch
from dappy import write
import numpy as np
import json
from tqdm import tqdm

babel = []
babel.append(json.load(open(HUMAN_DATA_PATH + "BABEL/val.json")))
babel.append(json.load(open(HUMAN_DATA_PATH + "BABEL/train.json")))
babel.append(json.load(open(HUMAN_DATA_PATH + "BABEL/test.json")))

splits = ["val", "train", "test"]

bm_fname = HUMAN_DATA_PATH + "babelmodel.npz"
num_betas = 16  # number of body parameters
meta = [[] for i in range(len(babel))]
data = [[] for i in range(len(babel))]
labels = [[] for i in range(len(babel))]

for datasplit in range(len(babel) - 1):
    for k in tqdm(babel[datasplit].keys()):
        meta[datasplit].append(
            [
                k,
                datasplit,
                babel[datasplit][k]["feat_p"].split("/")[0],
                babel[datasplit][k]["url"],
            ]
        )  # id, babel id, is validation set?, dataset ID, url to video
        amass_npz_fname = HUMAN_DATA_PATH + "AMASS/" + babel[datasplit][k]["feat_p"]
        bdata = np.load(amass_npz_fname)
        time_length = len(bdata["trans"])

        body_parms = {
            "root_orient": torch.Tensor(
                bdata["poses"][:, :3]
            ),  # controls the global root orientation
            "pose_body": torch.Tensor(bdata["poses"][:, 3:66]),  # controls the body
            "pose_hand": torch.Tensor(
                bdata["poses"][:, 66:]
            ),  # controls the finger articulation
            "trans": torch.Tensor(bdata["trans"]),  # controls the global body position
            "betas": torch.Tensor(
                np.repeat(
                    bdata["betas"][:num_betas][np.newaxis], repeats=time_length, axis=0
                )
            ),
        }

        bm = BodyModel(bm_path=bm_fname, num_betas=num_betas, model_type="smplh")

        bm.root_orient = torch.nn.Parameter(body_parms["root_orient"])
        bm.pose_body = torch.nn.Parameter(body_parms["pose_body"])
        bm.betas = torch.nn.Parameter(body_parms["betas"])
        bm.pose_hand = torch.nn.Parameter(body_parms["pose_hand"])

        body_pose_beta = bm(
            **{k: v for k, v in body_parms.items() if k in ["pose_body", "betas"]}
        )

        data[datasplit].append(np.array(body_pose_beta.Jtr.detach()))
        root_trans = np.repeat(
            bdata["trans"][:, None, ...], len(data[datasplit][0][0]), axis=1
        )
        data[datasplit][-1] += root_trans

        if babel[datasplit][k]["frame_ann"] == None:
            labels[datasplit].append(
                babel[datasplit][k]["seq_ann"]["labels"]
            )  # gives dict of act labels if no frame_ann
        else:
            seq = [
                [i["act_cat"], i["start_t"], i["end_t"]]
                for i in babel[datasplit][k]["frame_ann"]["labels"]
            ]  # these 3 lines give ordered list of frame labels for lists of actions
            sorter = np.array([i[1:] for i in seq])
            sortedseq = [seq[i] for i in np.argsort(sorter[:, 0])]
            framelabels = [[] for i in range(len(data[datasplit][-1]))]
            for segment in sortedseq:
                for i in range(
                    int(segment[1] * 120), min(int(segment[2] * 120), len(framelabels))
                ):
                    framelabels[i] += segment[0]
            labels[datasplit].append(framelabels)

ids = [[], []]
valframes = sum([len(data[0][i]) for i in range(len(data[0]))])
trainframes = sum([len(data[1][i]) for i in range(len(data[1]))])
ids[0] = np.concatenate([np.full(valframes, 0), np.full(trainframes, 1)])
data = np.concatenate(data)
ids[1] = np.concatenate([np.full(len(data[i]), i) for i in range(len(data))])
data = np.concatenate(data)
write.pose_h5(data, ids, HUMAN_DATA_PATH + "BABEL/babeldata_trans.h5")
meta = sum(meta, [])
np.save(
    HUMAN_DATA_PATH + "BABEL/babelmeta_trans.npy",
    np.asarray(meta, dtype="object"),
    allow_pickle=True,
)
labels = np.concatenate(labels)
labels = sum(labels, [])
np.save(
    HUMAN_DATA_PATH + "BABEL/babellabels_trans.npy",
    np.asarray(labels, dtype="object"),
    allow_pickle=True,
)
