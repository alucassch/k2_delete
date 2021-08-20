import k2
import torch
import logging
import k2.ragged as k2r

from typing import List


# Copied from PR https://github.com/espnet/espnet/pull/3466
def compile_HLG(
    L: k2.Fsa,
    G: k2.Fsa,
    H: k2.Fsa,
    labels_disambig_id_start: int,
    aux_labels_disambig_id_start: int,
) -> k2.Fsa:
    """Creates a decoding graph using a lexicon fst ``L`` and language model fsa ``G``.

    Involves arc sorting, intersection, determinization,
    removal of disambiguation symbols and adding epsilon self-loops.

    Args:
        L:
            An ``Fsa`` that represents the lexicon (L), i.e. has phones as ``symbols``
                and words as ``aux_symbols``.
        G:
            An ``Fsa`` that represents the language model (G), i.e. it's an acceptor
            with words as ``symbols``.
        H:  An ``Fsa`` that represents a specific topology used to convert the network
            outputs to a sequence of phones.
            Typically, it's a CTC topology fst, in which when 0 appears on the left
            side, it represents the blank symbol; when it appears on the right side,
            it indicates an epsilon.
        labels_disambig_id_start:
            An integer ID corresponding to the first disambiguation symbol in the
            phonetic alphabet.
        aux_labels_disambig_id_start:
            An integer ID corresponding to the first disambiguation symbol in the
            words vocabulary.
    :return:
    """
    L = k2.arc_sort(L)
    G = k2.arc_sort(G)
    # Attach a new attribute `lm_scores` so that we can recover
    # the `am_scores` later.
    # The scores on an arc consists of two parts:
    #  scores = am_scores + lm_scores
    # NOTE: we assume that both kinds of scores are in log-space.
    G.lm_scores = G.scores.clone()

    logging.info("Intersecting L and G")
    LG = k2.compose(L, G)
    logging.info(f"LG shape = {LG.shape}")
    logging.info("Connecting L*G")
    LG = k2.connect(LG)
    logging.info(f"LG shape = {LG.shape}")
    logging.info("Determinizing L*G")
    LG = k2.determinize(LG)
    logging.info(f"LG shape = {LG.shape}")
    logging.info("Connecting det(L*G)")
    LG = k2.connect(LG)
    logging.info(f"LG shape = {LG.shape}")
    logging.info("Removing disambiguation symbols on L*G")
    LG.labels[LG.labels >= labels_disambig_id_start] = 0
    if isinstance(LG.aux_labels, torch.Tensor):
        LG.aux_labels[LG.aux_labels >= aux_labels_disambig_id_start] = 0
    else:
        LG.aux_labels.values()[
            LG.aux_labels.values() >= aux_labels_disambig_id_start
        ] = 0
    logging.info("Removing epsilons")
    LG = k2.remove_epsilon(LG)
    logging.info(f"LG shape = {LG.shape}")
    logging.info("Connecting rm-eps(det(L*G))")
    LG = k2.connect(LG)
    logging.info(f"LG shape = {LG.shape}")
    LG.aux_labels = k2.ragged.remove_values_eq(LG.aux_labels, 0)

    logging.info("Arc sorting LG")
    LG = k2.arc_sort(LG)

    logging.info("Composing ctc_topo LG")
    HLG = k2.compose(H, LG, inner_labels="phones")

    logging.info("Connecting LG")
    HLG = k2.connect(HLG)

    logging.info("Arc sorting LG")
    HLG = k2.arc_sort(HLG)
    logging.info(
        f"LG is arc sorted: {(HLG.properties & k2.fsa_properties.ARC_SORTED) != 0}"
    )

    return HLG


# Copied from icefall/decode.py
def _intersect_device(
    a_fsas: k2.Fsa,
    b_fsas: k2.Fsa,
    b_to_a_map: torch.Tensor,
    sorted_match_a: bool,
    batch_size: int = 50,
) -> k2.Fsa:
    """This is a wrapper of k2.intersect_device and its purpose is to split
    b_fsas into several batches and process each batch separately to avoid
    CUDA OOM error.
    The arguments and return value of this function are the same as
    k2.intersect_device.
    """
    num_fsas = b_fsas.shape[0]
    if num_fsas <= batch_size:
        return k2.intersect_device(
            a_fsas, b_fsas, b_to_a_map=b_to_a_map, sorted_match_a=sorted_match_a
        )

    num_batches = (num_fsas + batch_size - 1) // batch_size
    splits = []
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_fsas)
        splits.append((start, end))

    ans = []
    for start, end in splits:
        indexes = torch.arange(start, end).to(b_to_a_map)

        fsas = k2.index(b_fsas, indexes)
        b_to_a = k2.index(b_to_a_map, indexes)
        path_lattice = k2.intersect_device(
            a_fsas, fsas, b_to_a_map=b_to_a, sorted_match_a=sorted_match_a
        )
        ans.append(path_lattice)

    return k2.cat(ans)


# Copied from icefall/decode.py
def nbest_decoding(
    lattice: k2.Fsa, num_paths: int, use_double_scores: bool = True
) -> k2.Fsa:
    """It implements something like CTC prefix beam search using n-best lists.
    The basic idea is to first extra n-best paths from the given lattice,
    build a word seqs from these paths, and compute the total scores
    of these sequences in the log-semiring. The one with the max score
    is used as the decoding output.
    Caution:
      Don't be confused by `best` in the name `n-best`. Paths are selected
      randomly, not by ranking their scores.
    Args:
      lattice:
        The decoding lattice, returned by :func:`get_lattice`.
      num_paths:
        It specifies the size `n` in n-best. Note: Paths are selected randomly
        and those containing identical word sequences are remove dand only one
        of them is kept.
      use_double_scores:
        True to use double precision floating point in the computation.
        False to use single precision.
    Returns:
      An FsaVec containing linear FSAs.
    """
    # First, extract `num_paths` paths for each sequence.
    # path is a k2.RaggedInt with axes [seq][path][arc_pos]
    path = k2.random_paths(lattice, num_paths=num_paths, use_double_scores=True)

    # word_seq is a k2.RaggedInt sharing the same shape as `path`
    # but it contains word IDs. Note that it also contains 0s and -1s.
    # The last entry in each sublist is -1.
    word_seq = k2.index(lattice.aux_labels, path)
    # Note: the above operation supports also the case when
    # lattice.aux_labels is a ragged tensor. In that case,
    # `remove_axis=True` is used inside the pybind11 binding code,
    # so the resulting `word_seq` still has 3 axes, like `path`.
    # The 3 axes are [seq][path][word_id]

    # Remove 0 (epsilon) and -1 from word_seq
    word_seq = k2.ragged.remove_values_leq(word_seq, 0)

    # Remove sequences with identical word sequences.
    #
    # k2.ragged.unique_sequences will reorder paths within a seq.
    # `new2old` is a 1-D torch.Tensor mapping from the output path index
    # to the input path index.
    # new2old.numel() == unique_word_seqs.tot_size(1)
    unique_word_seq, _, new2old = k2.ragged.unique_sequences(
        word_seq, need_num_repeats=False, need_new2old_indexes=True
    )
    # Note: unique_word_seq still has the same axes as word_seq

    seq_to_path_shape = k2.ragged.get_layer(unique_word_seq.shape(), 0)

    # path_to_seq_map is a 1-D torch.Tensor.
    # path_to_seq_map[i] is the seq to which the i-th path belongs
    path_to_seq_map = seq_to_path_shape.row_ids(1)

    # Remove the seq axis.
    # Now unique_word_seq has only two axes [path][word]
    unique_word_seq = k2.ragged.remove_axis(unique_word_seq, 0)

    # word_fsa is an FsaVec with axes [path][state][arc]
    word_fsa = k2.linear_fsa(unique_word_seq)

    # add epsilon self loops since we will use
    # k2.intersect_device, which treats epsilon as a normal symbol
    word_fsa_with_epsilon_loops = k2.add_epsilon_self_loops(word_fsa)

    # lattice has token IDs as labels and word IDs as aux_labels.
    # inv_lattice has word IDs as labels and token IDs as aux_labels
    inv_lattice = k2.invert(lattice)
    inv_lattice = k2.arc_sort(inv_lattice)

    path_lattice = _intersect_device(
        inv_lattice,
        word_fsa_with_epsilon_loops,
        b_to_a_map=path_to_seq_map,
        sorted_match_a=True,
    )
    # path_lat has word IDs as labels and token IDs as aux_labels

    path_lattice = k2.top_sort(k2.connect(path_lattice))

    tot_scores = path_lattice.get_tot_scores(
        use_double_scores=use_double_scores, log_semiring=False
    )

    # RaggedFloat currently supports float32 only.
    # If Ragged<double> is wrapped, we can use k2.RaggedDouble here
    ragged_tot_scores = k2.RaggedFloat(seq_to_path_shape, tot_scores.to(torch.float32))

    argmax_indexes = k2.ragged.argmax_per_sublist(ragged_tot_scores)

    # Since we invoked `k2.ragged.unique_sequences`, which reorders
    # the index from `path`, we use `new2old` here to convert argmax_indexes
    # to the indexes into `path`.
    #
    # Use k2.index here since argmax_indexes' dtype is torch.int32
    best_path_indexes = k2.index(new2old, argmax_indexes)

    path_2axes = k2.ragged.remove_axis(path, 0)

    # best_path is a k2.RaggedInt with 2 axes [path][arc_pos]
    best_path = k2.index(path_2axes, best_path_indexes)

    # labels is a k2.RaggedInt with 2 axes [path][token_id]
    # Note that it contains -1s.
    labels = k2.index(lattice.labels.contiguous(), best_path)

    labels = k2.ragged.remove_values_eq(labels, -1)

    # lattice.aux_labels is a k2.RaggedInt tensor with 2 axes, so
    # aux_labels is also a k2.RaggedInt with 2 axes
    aux_labels = k2.index(lattice.aux_labels, best_path.values())

    best_path_fsa = k2.linear_fsa(labels)
    best_path_fsa.aux_labels = aux_labels
    return best_path_fsa


# Copied from PR https://github.com/espnet/espnet/pull/3466
# Copied from: https://shorturl.at/diuCN
def find_first_disambig_symbol(symbols: k2.SymbolTable) -> int:
    return min(v for k, v in symbols._sym2id.items() if k.startswith("#"))


# Copied from PR https://github.com/espnet/espnet/pull/3466
# copied from:
# https://github.com/k2-fsa/snowfall/blob/master/snowfall/training/ctc_graph.py#L13
def build_ctc_topo(tokens: List[int]) -> k2.Fsa:
    """Build CTC topology.

    A token which appears once on the right side (i.e. olabels) may
    appear multiple times on the left side (ilabels), possibly with
    epsilons in between.
    When 0 appears on the left side, it represents the blank symbol;
    when it appears on the right side, it indicates an epsilon. That
    is, 0 has two meanings here.
    Args:
      tokens:
        A list of tokens, e.g., phones, characters, etc.
    Returns:
      Returns an FST that converts repeated tokens to a single token.
    """
    assert 0 in tokens, "We assume 0 is ID of the blank symbol"

    num_states = len(tokens)
    final_state = num_states
    arcs = ""
    for i in range(num_states):
        for j in range(num_states):
            if i == j:
                arcs += f"{i} {i} {tokens[i]} 0 0.0\n"
            else:
                arcs += f"{i} {j} {tokens[j]} {tokens[j]} 0.0\n"
        arcs += f"{i} {final_state} -1 -1 0.0\n"
    arcs += f"{final_state}"
    ans = k2.Fsa.from_str(arcs, num_aux_labels=1)
    ans = k2.arc_sort(ans)
    return ans


# Copied from PR https://github.com/espnet/espnet/pull/3466
# Modified from: https://github.com/k2-fsa/snowfall/blob/master/snowfall/common.py#L309
def get_texts(best_paths: k2.Fsa) -> List[List[int]]:
    """Extract the texts from the best-path FSAs.

     Args:
         best_paths:  a k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
                  containing multiple FSAs, which is expected to be the result
                  of k2.shortest_path (otherwise the returned values won't
                  be meaningful).  Must have the 'aux_labels' attribute, as
                a ragged tensor.
    Return:
        Returns a list of lists of int, containing the label sequences we
        decoded.
    """
    # remove any 0's or -1's (there should be no 0's left but may be -1's.)

    if isinstance(best_paths.aux_labels, k2.RaggedInt):
        aux_labels = k2r.remove_values_leq(best_paths.aux_labels, 0)
        aux_shape = k2r.compose_ragged_shapes(
            best_paths.arcs.shape(), aux_labels.shape()
        )

        # remove the states and arcs axes.
        aux_shape = k2r.remove_axis(aux_shape, 1)
        aux_shape = k2r.remove_axis(aux_shape, 1)
        aux_labels = k2.RaggedInt(aux_shape, aux_labels.values())
    else:
        # remove axis corresponding to states.
        aux_shape = k2r.remove_axis(best_paths.arcs.shape(), 1)
        aux_labels = k2.RaggedInt(aux_shape, best_paths.aux_labels)
        # remove 0's and -1's.
        aux_labels = k2r.remove_values_leq(aux_labels, 0)

    assert aux_labels.num_axes() == 2
    return k2r.to_list(aux_labels)
