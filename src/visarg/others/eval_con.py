import json
import argparse
import torch

def eval_con(gts, res, metrics=['blue:4']):
    """
    input:
    - gts: path to gts.json (expected to {'id': [gt1, gt2, ...], ...}) or dict
    - res: path to res.json (expected to {'id': [res], ...}) or dict
    - metrics: list of metrics (expected to ['bleu:n', 'cider', ...])
    
    output:
    - scores: dictionary of metrics and corresponding scores (ex. {'metric': score, ...})
    """
    
    # Read json files
    if isinstance(gts, str):
        with open(gts, 'r') as f:
            gts = json.load(f)
    if isinstance(res, str):
        with open(res, 'r') as f:
            res = json.load(f)
    
    # Prepare scorers
    scorers = []
    
    for metric in metrics:
        
        if 'bleu' in metric.lower():
            from visarg.others.pycocoevalcap.bleu.bleu import Bleu
            scorer, n = metric.split(':')
            
            scorers.append((Bleu(n=int(n)), scorer.lower()))
            
        elif 'cider' == metric.lower():
            from visarg.others.pycocoevalcap.cider.cider import Cider
            scorers.append((Cider(), metric.lower()))
            
        elif 'meteor' == metric.lower():
            from visarg.others.pycocoevalcap.meteor.meteor import Meteor
            scorers.append((Meteor(), metric.lower()))
            
        elif 'rouge' == metric.lower():
            from visarg.others.pycocoevalcap.rouge.rouge import Rouge
            scorers.append((Rouge(), metric.lower()))
            
        elif 'spice' == metric.lower():
            from visarg.others.pycocoevalcap.spice.spice import Spice
            scorers.append((Spice(), metric.lower()))
            
        elif 'bert' == metric.lower():
            from bert_score.scorer import BERTScorer
            scorers.append((BERTScorer(model_type="microsoft/deberta-xlarge-mnli", batch_size=32, lang="en", idf=True, idf_sents=[value[0] for value in gts.values()], rescale_with_baseline=True), metric.lower()))
    
    # Compute scores
    scores = {}
    for scorer in scorers:
        if scorer[1] == 'bert':
            res_strs = []
            gts_strs = []
            for key in res.keys():
                res_strs.append(res[key][0])
                gts_strs.append(gts[key][0])
            P, R, F1 = scorer[0].score(res_strs, gts_strs)
            scores[scorer[1]] = [sum(F1.tolist()) / len(F1.tolist()) * 100]
            continue
        score = scorer[0].compute_score(gts, res)
        scores[scorer[1]] = [score[0] * 100] if isinstance(score[0], float) else score
    
    del scorers
    
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gts_path', type=str)
    parser.add_argument('--res_path', type=str)
    parser.add_argument('--metrics', nargs='+', required=True, default=[])
    
    args = parser.parse_args()
    scores = eval_con(
        gts=args.gts_path,
        res=args.res_path,
        metrics=args.metrics
    )
    
    for metric, score in scores.items():
        print(f'{metric}:', score[0] if not isinstance(score[0], list) else [sc * 100 for sc in score[0]])