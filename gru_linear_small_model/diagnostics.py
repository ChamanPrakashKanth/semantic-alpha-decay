from __future__ import annotations

from typing import Dict

import torch


def complementarity(gru_pred: torch.Tensor, linear_pred: torch.Tensor, targets: torch.Tensor):
    gc, lc = gru_pred.eq(targets), linear_pred.eq(targets)
    both_correct = int((gc & lc).sum())
    both_wrong = int((~gc & ~lc).sum())
    g_only = int((gc & ~lc).sum())
    l_only = int((~gc & lc).sum())
    g_wrong = max(int((~gc).sum()), 1)
    l_wrong = max(int((~lc).sum()), 1)
    return {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "gru_correct_linear_wrong": g_only,
        "gru_wrong_linear_correct": l_only,
        "p_linear_correct_given_gru_wrong": l_only / g_wrong,
        "p_gru_correct_given_linear_wrong": g_only / l_wrong,
        "prediction_disagreement": float(gru_pred.ne(linear_pred).float().mean()),
        "oracle_accuracy": float((gc | lc).float().mean()),
    }


def rescue_damage(reference_pred, candidate_pred, targets):
    rc, cc = reference_pred.eq(targets), candidate_pred.eq(targets)
    p_ref = float(rc.float().mean())
    rescue = float((cc & ~rc).sum()) / max(int((~rc).sum()), 1)
    damage = float((~cc & rc).sum()) / max(int(rc.sum()), 1)
    delta = float(cc.float().mean() - rc.float().mean())
    identity = (1.0 - p_ref) * rescue - p_ref * damage
    return {
        "reference_accuracy": p_ref,
        "candidate_accuracy": float(cc.float().mean()),
        "rescue_rate": rescue,
        "damage_rate": damage,
        "delta_accuracy": delta,
        "identity_rhs": identity,
        "identity_error": abs(delta - identity),
    }


def fusion_diagnostics(lambdas, masks, roles, chain_lengths, gru_pred, linear_pred, targets):
    valid = masks.unsqueeze(-1).expand_as(lambdas)
    values = lambdas[valid]
    report = {
        "mean_lambda": float(values.mean()),
        "std_lambda": float(values.std(unbiased=False)),
        "median_lambda": float(values.median()),
        "fraction_lambda_lt_0_1": float((values < 0.1).float().mean()),
        "fraction_lambda_gt_0_9": float((values > 0.9).float().mean()),
    }
    report["expert_collapse"] = bool(
        (report["mean_lambda"] > 0.98 or report["mean_lambda"] < 0.02)
        and report["std_lambda"] < 0.02
    )
    by_timestep = {}
    for t in range(lambdas.shape[1]):
        active = masks[:, t]
        if active.any():
            by_timestep[str(t)] = float(lambdas[active, t].mean())
    report["lambda_by_timestep"] = by_timestep
    report["lambda_true_chain_tokens"] = float(lambdas[roles.eq(1)].mean())
    report["lambda_distractor_tokens"] = (
        float(lambdas[roles.eq(2)].mean()) if roles.eq(2).any() else None
    )
    report["lambda_by_chain_length"] = {
        str(int(length)): float(lambdas[chain_lengths.eq(length)][masks[chain_lengths.eq(length)]].mean())
        for length in chain_lengths.unique()
    }
    gc, lc = gru_pred.eq(targets), linear_pred.eq(targets)
    conditions = {
        "both_correct": gc & lc,
        "gru_correct_linear_wrong": gc & ~lc,
        "gru_wrong_linear_correct": ~gc & lc,
        "both_wrong": ~gc & ~lc,
    }
    conditioned = {}
    for name, rows in conditions.items():
        if rows.any():
            conditioned[name] = float(lambdas[rows][masks[rows]].mean())
        else:
            conditioned[name] = None
    report["lambda_by_branch_correctness"] = conditioned
    return report
