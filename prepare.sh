#!/bin/bash

set -eou pipefail

lang_dir=data/lang
lm=model_data/lm.arpa

base_url=http://www.linse.ufsc.br:/~andrelucas/k2

# Download data
wget -O data.tar.gz $base_url/data.tar.gz || exit 1

# sha1sum f884ed107d436108ca591b094b9c1eff20a421c3  data.tar.gz
tar -xvf data.tar.gz

# The same as espnet2.bin.extract_token_list
python3 << END
import yaml
with open("model_data/config.yaml") as f:
    args = yaml.safe_load(f)
    assert "token_list" in args
    token_list = args["token_list"]
    token_and_idx_lines = ""
    for token_idx, token in enumerate(token_list):
        token_and_idx_lines += f"{token} {token_idx}\n"

    with open("data/lang/tokens.txt","w") as fout:
        fout.write(token_and_idx_lines)
END

echo "<eps> 0" > $lang_dir/words.txt
# We use <unk> lowercase
echo "<unk> 1" >> $lang_dir/words.txt

echo "<unk> <unk>" > ${lang_dir}/lexicon.txt
cat ${lang_dir}/main_lexicon.lex >> ${lang_dir}/lexicon.txt

cat ${lang_dir}/main_lexicon.lex | awk '{print $1}' | sort -u  > ${lang_dir}/vocab.txt
cat ${lang_dir}/vocab.txt | awk '{print $1, FNR + 1}' >> ${lang_dir}/words.txt

perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' < $lang_dir/lexicon.txt > $lang_dir/lexiconp.txt

if ! grep "#0" $lang_dir/words.txt > /dev/null 2>&1; then
    max_word_id=$(tail -1 $lang_dir/words.txt | awk '{print $2}')
    echo "#0 $((max_word_id+1))" >> $lang_dir/words.txt
fi

ndisambig=$(utils/add_lex_disambig.pl --pron-probs $lang_dir/lexiconp.txt $lang_dir/lexiconp_disambig.txt)

if ! grep "#0" $lang_dir/tokens.txt > /dev/null 2>&1 ; then
    max_token_id=$(tail -1 $lang_dir/tokens.txt | awk '{print $2}')
    for i in $(seq 0 $ndisambig); do
        echo "#$i $((i+max_token_id+1))"
    done >> $lang_dir/tokens.txt
fi

if [ ! -f $lang_dir/L_disambig.fst.txt ]; then
    wdisambig_token=$(echo "#0" | utils/sym2int.pl $lang_dir/tokens.txt)
    wdisambig_word=$(echo "#0" | utils/sym2int.pl $lang_dir/words.txt)

    python3 local/make_lexicon_fst.py \
        $lang_dir/lexiconp_disambig.txt | \
        utils/sym2int.pl --map-oov 1 -f 3 $lang_dir/tokens.txt | \
        utils/sym2int.pl -f 4 $lang_dir/words.txt  | \
        local/fstaddselfloops.pl $wdisambig_token $wdisambig_word > $lang_dir/L_disambig.fst.txt || exit 1
fi

if [ ! -f $lang_dir/G.fst.txt ]; then
    python3 -m kaldilm \
        --read-symbol-table="$lang_dir/words.txt" \
        --disambig-symbol='#0' \
        --max-order=2 \
        ${lm} > $lang_dir/G.fst.txt

fi