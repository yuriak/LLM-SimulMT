import sys

import numpy as np
import random
import tqdm
import argparse
import json
import pickle


def get_missing_nodes(sample_src, sample_tgt, idx):
    missing_src_idx = sorted(list(set(list(range(len(sample_src.split())))) - set(np.array(idx)[:, 0])))
    missing_tgt_idx = sorted(list(set(list(range(len(sample_tgt.split())))) - set(np.array(idx)[:, 1])))
    return missing_src_idx, missing_tgt_idx


def fill_src_missing_nodes(missing_src_idx, contineous_idx):
    for x in missing_src_idx:
        idx_arr = np.array(contineous_idx)
        # inserted = False
        # while not inserted:
        if x == 0:
            _, y_prev, _ = contineous_idx[0]
            contineous_idx.insert(0, [0, y_prev, 1])
            # inserted = True
        else:
            x_prev = x - 1
            insert_pos = (idx_arr[:, 0] == x_prev).cumsum().argmax()
            y_prev = idx_arr[insert_pos, 1].tolist()
            contineous_idx.insert(insert_pos + 1, [x, y_prev, 1])
            # inserted = True
    return contineous_idx


def fill_tgt_missing_nodes(missing_tgt_idx, contineous_idx):
    for x in missing_tgt_idx:
        idx_arr = np.array(contineous_idx)
        inserted = False
        move_step = 1
        while not inserted:
            if x + move_step in idx_arr[:, 1]:
                insert_pos = (idx_arr[:, 1] == x + 1).argmax()
                sidx, tidx, _ = contineous_idx[insert_pos]
                contineous_idx.insert(insert_pos, [sidx, x, -1])
                idx_arr = np.array(contineous_idx)
                inserted = True
            elif x + move_step >= idx_arr[:, 1].max():
                insert_pos = -1
                sidx, tidx, _ = contineous_idx[insert_pos]
                contineous_idx.append([sidx, x, -1])
                inserted = True
            else:
                move_step += 1
                continue
    return contineous_idx


def prepare_alignment(sample_src, sample_tgt, alignment):
    idx = [[int(i) for i in x.split('-')] for x in alignment.split()]
    missing_src_idx, missing_tgt_idx = get_missing_nodes(sample_src, sample_tgt, idx)
    contineous_idx = [[x, y, 0] for x, y in idx]
    contineous_idx = fill_src_missing_nodes(missing_src_idx, contineous_idx)
    contineous_idx = fill_tgt_missing_nodes(missing_tgt_idx, contineous_idx)
    contineous_idx = np.array(contineous_idx)[:, :2]
    return sorted(contineous_idx.tolist(), key=lambda x: x[1])


def create_graph(sample_src, sample_tgt, alignment_idx):
    X_inner_edges = [[x - 1, x] for x in range(1, len(sample_src.split()))]
    Y_inner_edges = [[x - 1, x] for x in
                     range(len(sample_src.split()) + 1, len(sample_src.split()) + len(sample_tgt.split()))]

    relation_edges = sorted(alignment_idx, key=lambda x: x[0])
    relation_edges = [[x[0], x[1] + len(sample_src.split())] for x in relation_edges]
    #     Create dependency graph
    x_graph = {}

    if len(X_inner_edges) == 0:
        x_graph[0] = []
    else:
        for x0, x1 in X_inner_edges:
            if x0 not in x_graph:
                x_graph[x0] = []
            x_graph[x0].append(x1)

    for x, y in relation_edges:
        if x not in x_graph:
            x_graph[x] = []
        x_graph[x].append(y)

    y_graph = {}
    if len(Y_inner_edges) > 0:
        y_graph[Y_inner_edges[0][0]] = []
        for y0, y1 in Y_inner_edges:
            y_graph[y1] = [y0]
    else:
        y_initial = min([x[1] for x in relation_edges])
        y_graph[y_initial] = []

    for x, y in relation_edges:
        if y not in y_graph:
            y_graph[y] = []
        y_graph[y].append(x)

    #     Add necessary edges, fix temporal reordering
    for i, k in enumerate(y_graph.keys()):
        if i == 0:
            continue
        inner_causal = y_graph[k][0]
        max_x_causal = y_graph[k][-1]
        previous_max_x_causal = y_graph[inner_causal][-1]
        if previous_max_x_causal > max_x_causal:
            y_graph[k].append(previous_max_x_causal)

    #     Recreate X graph
    for i, (k, v) in enumerate(y_graph.items()):
        if i == 0:
            continue
        inner_causal = y_graph[k][0]
        x_causal = y_graph[k][1:]
        for x in x_causal:
            if k not in x_graph[x]:
                x_graph[x].append(k)
    return x_graph, y_graph


def basic_policy(graph):
    seq = []
    y_cursor = 0
    x_cursor = 0

    if len(graph) > 1:
        for i, (k, v) in enumerate(graph.items()):
            if i == 0:
                # if v[0]>0:
                seq += [('R', [j for j in range(x_cursor, v[0] + 1)])]
                x_cursor = v[0]
                # seq +=[('R',[v[0]])]
                y_cursor = k
                continue
            y_dep, x_deps = v[0], v[1:]
            if k == y_dep + 1:
                seq += [('W', [y_dep])]
                y_cursor = k
            if x_deps[-1] > x_cursor:
                seq += [('R', list(range(x_cursor + 1, x_deps[-1] + 1)))]
                x_cursor = x_deps[-1]
    else:
        k, v = tuple(graph.items())[0]
        seq += [('R', v)]
    seq.append(('W', [k]))

    merged_seq = []
    for a, s in seq:
        if len(merged_seq) > 0:
            pa, ps = merged_seq[-1]
            if pa == a:
                merged_seq[-1] = (pa, ps + s)
            else:
                merged_seq.append((a, s))
        else:
            merged_seq.append((a, s))
    return merged_seq


def merge_seq(seq, read_n=3):
    read_buffer = []
    write_buffer = []
    merged_seq = []
    for i, (a, segs) in enumerate(seq):
        if a == 'R':
            if len(segs) < read_n:
                if len(read_buffer) + len(segs) <= read_n:
                    read_buffer = read_buffer + segs
                else:
                    merged_seq += [('R', read_buffer)]
                    if len(write_buffer) > 0:
                        merged_seq += [('W', write_buffer)]
                        write_buffer = []
                    read_buffer = segs
            else:
                merged_seq += [('R', read_buffer)]
                if len(write_buffer) > 0:
                    merged_seq += [('W', write_buffer)]
                    write_buffer = []
                read_buffer = segs
        else:
            write_buffer += segs
    if len(read_buffer) > 0:
        merged_seq += [('R', read_buffer)]
    if len(write_buffer) > 0:
        merged_seq += [('W', write_buffer)]
    return merged_seq


def dynamic_merge_seq(seq, min_continues_read=3, max_min_continues_read=8):
    read_buffer = []
    write_buffer = []
    merged_seq = []
    for i, (a, segs) in enumerate(seq):
        read_count = random.randint(min_continues_read, max_min_continues_read)
        if a == 'R':
            if len(segs) < read_count:
                if len(read_buffer) + len(segs) <= read_count:
                    read_buffer = read_buffer + segs
                else:
                    merged_seq += [('R', read_buffer)]
                    if len(write_buffer) > 0:
                        merged_seq += [('W', write_buffer)]
                        write_buffer = []
                    read_buffer = segs
            else:
                merged_seq += [('R', read_buffer)]
                if len(write_buffer) > 0:
                    merged_seq += [('W', write_buffer)]
                    write_buffer = []
                read_buffer = segs
        else:
            write_buffer += segs
    if len(read_buffer) > 0:
        merged_seq += [('R', read_buffer)]
    if len(write_buffer) > 0:
        merged_seq += [('W', write_buffer)]
    return merged_seq


def to_text(sample_src, sample_tgt, seq):
    offset = list(filter(lambda x: x[0] == 'W', seq))[0][1][0]
    src_tokens = sample_src.split()
    tgt_tokens = sample_tgt.split()
    return [(a, ' '.join([src_tokens[w] if a == 'R' else tgt_tokens[w - offset] for w in x]).strip()) for a, x in seq]


def generate_trajectory(sample_src, sample_tgt, alignment, read_n=3):
    alignment_idx = prepare_alignment(sample_src, sample_tgt, alignment)
    x_graph, y_graph = create_graph(sample_src, sample_tgt, alignment_idx)
    seq = basic_policy(y_graph)
    if type(read_n) == str and 'random' in read_n:
        _, s, e = read_n.split(':')
        # read_n = random.randint(,)
        seq = dynamic_merge_seq(seq, min_continues_read=int(s), max_min_continues_read=int(e))
    else:
        if read_n > 1:
            seq = merge_seq(seq, read_n=read_n)
    seq = to_text(sample_src, sample_tgt, seq)
    return seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="src path")
    parser.add_argument("--tgt", help="tgt path")
    parser.add_argument("--src-lang", help="source lang")
    parser.add_argument("--tgt-lang", help="target lang")
    parser.add_argument("--alignment", help="alignment path")
    parser.add_argument("--forward", help="forward path", required=False, default=None)
    parser.add_argument("--reverse", help="reverse path", required=False, default=None)
    parser.add_argument("--output", help="output path")
    parser.add_argument("--output-pickle", help="output pickle path")
    parser.add_argument("--filter-threshold", help="clean noisy samples", required=False, default=0.5)
    args = parser.parse_args()
    src = [x.strip() for x in open(args.src, 'r').readlines()]
    tgt = [x.strip() for x in open(args.tgt, 'r').readlines()]
    alignment = [x.strip() for x in open(args.alignment, 'r').readlines()]
    filter_threshold = args.filter_threshold
    do_filter = False
    if None not in [args.forward, args.reverse]:
        do_filter = True
        forward = [x.strip() for x in open(args.forward, 'r').readlines()]
        reverse = [x.strip() for x in open(args.reverse, 'r').readlines()]

    sequences = []

    for i, (x, y, a) in tqdm.tqdm(enumerate(zip(src, tgt, alignment)), total=len(src)):
        if do_filter:
            fa = forward[i].split()
            ra = reverse[i].split()
            score = sum(pair in ra for pair in fa) / max(len(ra), len(fa))
            if score < filter_threshold:
                continue
        result = generate_trajectory(x, y, a, read_n=-1)
        s = ' '.join(list(map(lambda x: x[1], list(filter(lambda x: x[0] == 'R', result))))).strip()
        t = ' '.join(list(map(lambda x: x[1], list(filter(lambda x: x[0] == 'W', result))))).strip()
        assert s == src[i].strip(), i
        assert t == tgt[i].strip(), i
        sequences.append(result)

    print(f"{len(sequences)}/{len(src)}={len(sequences) / len(src)} preserved")

    seq_json = [json.dumps(x, ensure_ascii=False) for x in sequences]
    open(args.output, 'w').write('\n'.join(seq_json) + '\n')

    sequences = [{
        "text": seq,
        "lang": f'{args.src_lang}-{args.tgt_lang}',
        "input_len": len(' '.join(list(map(lambda x: x[1], seq))).split())
    } for seq in sequences]

    pickle.dump(sequences, open(args.output_pickle, 'wb'))

    print("done")


if __name__ == '__main__':
    main()
