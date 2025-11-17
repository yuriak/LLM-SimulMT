# import sys
# sys.path.append('./')

from argparse import Namespace, ArgumentParser
import time
import tqdm
import json
import os
import numpy as np
import torch
import re
import jieba


pack_to_device = lambda x, d: {k: v.cuda(d) for k, v in x.items()}


def ReadAction():
    return False, None, None


def WriteAction(content, finished):
    return True, content, finished


class AgentStates:
    """
    Tracker of the decoding progress.

    Attributes:
        source (list): current source sequence.
        target (list): current target sequence.
        source_finished (bool): if the source is finished.
        target_finished (bool): if the target is finished.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset Agent states"""
        self.source = []
        self.target = []
        self.source_finished = False
        self.target_finished = False
        self.rw_record = []

    def update_source(self, segment, segment_tokenizer):
        """
        Update states from input segment

        Args:
            segment (~simuleval.agents.segments.Segment): input segment
        """
        if segment is not None and len(segment) > 0:
            self.source.append(segment)
            self.rw_record.append(([0] * len(segment_tokenizer(segment)), segment))

    def update_target(self, segment, time_elapse, segment_tokenizer):
        """
        Update states from output segment

        Args:
            segment (~simuleval.agents.segments.Segment): input segment
        """
        if segment is not None and len(segment) > 0:
            self.target.append(segment)
            self.rw_record.append(
                ([1] * len(segment_tokenizer(segment)), segment, time_elapse)
            )


class Agent:
    def __init__(self, args):
        self.states = AgentStates()

    def policy(self):
        pass

    def reset(self):
        self.states.reset()


class Evaluator:

    def __init__(self, args, agent: Agent):
        self.args = args
        self.head_n = args.head_n
        self.source = [x.strip() for x in open(args.src, "r").readlines()]
        self.target = [x.strip() for x in open(args.tgt, "r").readlines()]

        if self.head_n is not None:
            print(f"Head {self.head_n}")
            self.source = self.source[: self.head_n]
            self.target = self.target[: self.head_n]

        assert len(self.source) == len(self.target)
        os.makedirs(args.output, exist_ok=True)
        self.output_path = args.output
        self.agent = agent
        self.tokenizers = {
            "space": lambda x: x.split(),
            "char": lambda x: list(x),
            "jieba": lambda x: [x.strip() for x in jieba.lcut(x) if len(x.strip()) > 0],
        }
        self.src_tokenizer = self.tokenizers.get(
            args.src_tokenizer, lambda x: x.split()
        )
        self.tgt_tokenizer = self.tokenizers.get(
            args.tgt_tokenizer, lambda x: x.split()
        )

        self.src_chunk_sep = args.src_chunk_sep
        self.tgt_chunk_sep = args.tgt_chunk_sep

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("--src", type=str, required=True)
        parser.add_argument("--tgt", type=str, required=True)
        parser.add_argument("--head-n", type=int, required=False)
        parser.add_argument("--output", type=str, required=True)
        parser.add_argument(
            "--src_tokenizer", type=str, default="space", required=False
        )
        parser.add_argument(
            "--tgt_tokenizer", type=str, default="space", required=False
        )
        parser.add_argument("--src_chunk_sep", type=str, default=" ", required=False)
        parser.add_argument("--tgt_chunk_sep", type=str, default=" ", required=False)

    def inference_instance(self, src, tgt):
        tokens = self.src_tokenizer(src)
        self.agent.reset()
        # self.agent.states.update_source(tokens.pop(0))
        self.agent.states.source_finished = len(tokens) == 0
        sample_start_time = time.time()
        while not self.agent.states.target_finished:
            if not self.agent.states.source_finished:
                self.agent.states.update_source(tokens.pop(0), self.src_tokenizer)
                self.agent.states.source_finished = len(tokens) == 0
            policy_start_time = time.time()
            write, content, target_finished = self.agent.policy()
            policy_time = time.time() - policy_start_time

            if write and content is not None:
                self.agent.states.update_target(content, policy_time, self.tgt_tokenizer)
            self.agent.states.target_finished = target_finished
        record = {
            "src": self.src_chunk_sep.join(self.agent.states.source),
            "hyp": self.tgt_chunk_sep.join(self.agent.states.target),
            "tgt": tgt,
            "rw": self.agent.states.rw_record,
            "time": time.time() - sample_start_time,
        }
        return record

    def evaluate(self):
        print("Start evaluating....")
        records = []
        for src, tgt in tqdm.tqdm(list(zip(self.source, self.target))):
            record = self.inference_instance(src, tgt)
            records.append(record)
        return records

    def dump_results(self, results):
        json.dump(results, open(self.output_path + "/results.json", "w"))

    def score_quality(self, results):
        import sacrebleu

        bleu = sacrebleu.corpus_bleu(
            [x["hyp"] for x in results], [[x["tgt"] for x in results]],
            tokenize="zh" if self.args.tgt_tokenizer == "jieba" else "13a",
        ).score

        from huggingface_hub import hf_hub_download
        from comet import download_model, load_from_checkpoint

        model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(model_path)
        comet_inputs = [
            {"src": x["src"], "mt": x["hyp"], "ref": x["tgt"]} for x in results
        ]
        comet_output = comet_model.predict(comet_inputs, batch_size=32, gpus=1)

        return {"BLEU": bleu, "COMET": comet_output.system_score}, comet_output.scores

    def score_latency(self, results):
        def rw_to_delay(rws):
            delays = []
            consuctive_read = 0
            for a in rws:
                if a[0] == 0:
                    consuctive_read += len(a)
                if a[0] == 1:
                    delays.extend([consuctive_read] * len(a))
            return delays

        for i in range(len(results)):
            sample = results[i]
            sample["delays"] = rw_to_delay([x[0] for x in sample["rw"]])
            sample["source_length"] = len(self.src_tokenizer(sample["src"]))
            sample["target_length"] = len(self.tgt_tokenizer(sample["tgt"]))

        def compute_token_wall_time(rw=None, **kwargs):
            if len(rw) == 0:
                return 0

            mean_time_per_token = np.mean(
                list(
                    map(
                        lambda u: (
                            u[-1] / len(u[0]) * 1000 if len(u[0]) > 0 else u[-1] * 1000
                        ),
                        filter(lambda x: len(x) == 3, rw),
                    )
                )
            )

            return mean_time_per_token

        def compute_al(delays=None, source_length=None, target_length=None, **kwargs):

            if delays[0] > source_length:
                return delays[0]

            AL = 0
            gamma = target_length / source_length
            tau = 0
            for t_minus_1, d in enumerate(delays):
                AL += d - t_minus_1 / gamma
                tau = t_minus_1 + 1

                if d >= source_length:
                    break
            AL /= tau
            return AL

        def compute_laal(delays=None, source_length=None, target_length=None, **kwargs):
            if delays[0] > source_length:
                return delays[0]

            LAAL = 0
            gamma = max(len(delays), target_length) / source_length
            tau = 0
            for t_minus_1, d in enumerate(delays):
                LAAL += d - t_minus_1 / gamma
                tau = t_minus_1 + 1

                if d >= source_length:
                    break
            LAAL /= tau
            return LAAL

        def compute_ap(delays=None, source_length=None, target_length=None, **kwargs):
            return sum(delays) / (source_length * target_length)

        def compute_dal(delays=None, source_length=None, target_length=None, **kwargs):
            DAL = 0
            target_length = len(delays)
            gamma = target_length / source_length
            g_prime_last = 0
            for i_minus_1, g in enumerate(delays):
                if i_minus_1 + 1 == 1:
                    g_prime = g
                else:
                    g_prime = max([g, g_prime_last + 1 / gamma])

                DAL += g_prime - i_minus_1 / gamma
                g_prime_last = g_prime

            DAL /= target_length
            return DAL

        scores = {}
        for metric_name, compute in zip(
            ["AL", "LAAL", "AP", "DAL", "TWT"],
            [
                compute_al,
                compute_laal,
                compute_ap,
                compute_dal,
                compute_token_wall_time,
            ],
        ):
            scores[metric_name] = np.mean([compute(**instance) for instance in results])
        return scores

    def compute_performance(self, results):
        import pandas as pd

        q_perf, instance_comet = self.score_quality(results)
        l_perf = self.score_latency(results)
        gpu_type = torch.cuda.get_device_name(getattr(self.args, "cuda", 0))
        print("GPU:", gpu_type)
        final_result = {}
        for k, v in q_perf.items():
            final_result[k] = v
        for k, v in l_perf.items():
            final_result[k] = v
        final_result["GPU"] = str(gpu_type)

        df = pd.DataFrame(final_result, index=[0])
        df.to_csv(
            self.output_path + "/scores.csv",
            index=None,
            sep="\t",
            float_format="{:.3f}".format,
        )
        print(df)

        with open(self.output_path + "/comet_instance.txt", "w") as f:
            for item in instance_comet:
                f.write("%s\n" % item)
        return df.to_dict(orient="records")[0]


#
# if __name__ == '__main__':
#     parser = ArgumentParser()
#     Evaluator.add_args(parser)
#     MTurnLcpAgent.add_args(parser)
#     args = parser.parse_args()
#     evaluator = Evaluator(args)
#     results = evaluator.evaluate()
#     evaluator.dump_results(results)
#     evaluator.compute_performance(results)
