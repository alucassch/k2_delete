import os
import k2
import torch
import librosa
import numpy as np
from pathlib import Path
from espnet2.tasks.asr import ASRTask

from helpers import *


def get_test_data():
    audio = {}
    text = {}
    with open("audio/wav.scp") as faudio, open("audio/text") as ftext:
        for line in faudio:
            uttid, audiofile = line.strip().split()
            audio[uttid] = audiofile

        for line in ftext:
            uttid, sentence = line.strip().split(None, 1)
            text[uttid] = sentence

    assert audio.keys() == text.keys()

    for uttid in audio:
        yield uttid, audio[uttid], text[uttid]


if __name__ == "__main__":

    asr_train_config = "model_data/config.yaml"
    asr_model_file = "model_data/48epoch.pth"

    asr_model, asr_train_args = ASRTask.build_model_from_file(
        asr_train_config, asr_model_file, "cpu"
    )

    asr_model.to(dtype=getattr(torch, "float32")).eval()
    lang_dir = Path("data/lang")

    symbol_table = k2.SymbolTable.from_file(lang_dir / "words.txt")
    token_symbol_table = k2.SymbolTable.from_file(lang_dir / "tokens.txt")
    token_list = asr_model.token_list

    token_ids = {i: token for i, token in enumerate(token_list)}

    if not os.path.exists(lang_dir / "HLG.pt"):
        print("Creating HLG.pt...")

        ctc_topo = k2.arc_sort(build_ctc_topo(list(range(len(token_list)))))

        with open(lang_dir / "L_disambig.fst.txt") as f:
            L = k2.Fsa.from_openfst(f.read(), acceptor=False)

        with open(lang_dir / "G.fst.txt") as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)

        first_token_disambig_id = find_first_disambig_symbol(token_symbol_table)
        first_word_disambig_id = find_first_disambig_symbol(symbol_table)

        HLG = compile_HLG(
            L=L,
            G=G,
            H=ctc_topo,
            labels_disambig_id_start=first_token_disambig_id,
            aux_labels_disambig_id_start=first_word_disambig_id,
        )

        torch.save(HLG.as_dict(), lang_dir / "HLG.pt")
    else:
        d = torch.load(lang_dir / "HLG.pt")
        HLG = k2.Fsa.from_dict(d)

    for uttid, audiofile, text in get_test_data():

        samples, _ = librosa.load(audiofile, sr=16000)

        speech = torch.tensor(samples).unsqueeze(0).to(getattr(torch, "float32"))
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))

        with torch.no_grad():
            hs_pad, _ = asr_model.encode(speech, lengths)
            lpz = torch.nn.functional.log_softmax(asr_model.ctc.ctc_lo(hs_pad), dim=2)

        supervision_segments = torch.tensor(
            [[0, 0, hs_pad.shape[1]]], dtype=torch.int32
        )

        dense_fsa_vec = k2.DenseFsaVec(lpz, supervision_segments)
        lattices = k2.intersect_dense_pruned(HLG, dense_fsa_vec, 20.0, 20, 30, 10000)

        print(uttid)
        print("GROUND TRUTH:", text)
        # Get k2 shortest path words result
        shortest_path_fsa = k2.shortest_path(lattices, use_double_scores=True)
        hyp_shortest_path = get_texts(shortest_path_fsa)
        hyp_shortest_path = " ".join(
            [symbol_table.get(i) for i in hyp_shortest_path[0]]
        )
        print("K2 SHORTPATH:", hyp_shortest_path)

        # # Get k2 nbest words result
        # best_path_fsa = nbest_decoding(lattices, 10)
        # hyp_best_path = get_texts(best_path_fsa)
        # hyp_best_path = " ".join([symbol_table.get(i) for i in hyp_best_path[0]])
        # print("K2 NBESTPATH:", hyp_best_path + "\n")

        # Do CTC greedy search to check for high score CTC symbols
        # CTC tokens are phones subwords
        ctc_prob = lpz.exp_().numpy()
        ctc_tokens = [
            token_ids[np.argmax(ctc_prob[0][i, :])] for i in range(lpz.shape[1])
        ]
        ctc_scores = [max(ctc_prob[0][i, :]) for i in range(lpz.shape[1])]
        # Remove <blank> token
        ctc_tokens = [x for x in zip(ctc_tokens, ctc_scores) if x[0] != token_ids[0]]
        print("CTC GREEDY  :", ctc_tokens, "\n")

