from sentence_transformers import SentenceTransformer
import argparse
import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument("--src")
parser.add_argument("--tgt")
parser.add_argument("--output-path")
parser.add_argument("--cuda", default='0:4')
parser.add_argument("--threshold", default=0.7, type=float)
parser.add_argument("--bsz", default=256, type=int)

args = parser.parse_args()
output_path = args.output_path
os.makedirs(output_path, exist_ok=True)

cuda_id, num_gpu = list(map(lambda x: int(x), args.cuda.split(':')))
src = list(map(lambda x: x.strip(), open(args.src, 'r').readlines()))
tgt = list(map(lambda x: x.strip(), open(args.tgt, 'r').readlines()))
assert len(src) == len(tgt)

shard_size = len(src) // num_gpu + 1
src = src[cuda_id * shard_size:(cuda_id + 1) * shard_size]
tgt = tgt[cuda_id * shard_size:(cuda_id + 1) * shard_size]

model = SentenceTransformer('sentence-transformers/LaBSE', device=f'cuda:{cuda_id}')


def get_score(srcs, tgts):
    s_embeddings = model.encode(srcs, device=f'cuda:{cuda_id}')
    t_embeddings = model.encode(tgts, device=f'cuda:{cuda_id}')
    return (s_embeddings * t_embeddings).sum(-1).tolist()


bsz = args.bsz
batches = len(src) // bsz + 1
filtered_src = []
filtered_tgt = []

for i in tqdm.tqdm(range(batches)):
    bsrc = src[i * bsz:(i + 1) * bsz]
    btgt = tgt[i * bsz: (i + 1) * bsz]
    scores = get_score(bsrc, btgt)
    filtered_result = list(filter(lambda x: x[2] > args.threshold, zip(bsrc, btgt, scores)))
    if len(filtered_result) > 0:
        fsrc, ftgt, _ = list(zip(*filtered_result))
        filtered_src.extend(list(fsrc))
        filtered_tgt.extend(list(ftgt))

open(os.path.join(output_path, f'{cuda_id}.src'), 'w').write('\n'.join(filtered_src) + '\n')
open(os.path.join(output_path, f'{cuda_id}.tgt'), 'w').write('\n'.join(filtered_tgt) + '\n')

print(f"LabSE filter completed for {cuda_id}")